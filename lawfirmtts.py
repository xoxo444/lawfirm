import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from dotenv import load_dotenv
from docx import Document
import google.generativeai as genai
import numpy as np
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")  # Add your key in .env file as GOOGLE_API_KEY=your_key_here
genai.configure(api_key=api_key)

gemini = genai.GenerativeModel("gemini-1.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#loading the legal cases
def load_cases(folder_path="./cases"):  # Place .docx files in a 'cases' folder
    cases = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".docx") and filename.lower().startswith("case"):
            file_path = os.path.join(folder_path, filename)
            document = Document(file_path)
            current_case = {"source": filename, "summaries": "", "charges": []}
            summary_lines = []

            for para in document.paragraphs:
                text = para.text.strip()
                if not text:
                    continue
                if " v. " in text and "client" not in current_case:
                    current_case["client"] = text
                elif "section" in text.lower() and not current_case["charges"]:
                    charges = text.split(":", 1)[-1].strip()
                    current_case["charges"] = [s.strip() for s in charges.split(",")]
                else:
                    summary_lines.append(text)

            current_case["summaries"] = "\n".join(summary_lines).strip()
            if "client" in current_case:
                cases.append(current_case)
    return cases

#search by client name
def case_assistant(client_name, question=""):
    cases = load_cases()
    if not client_name or not cases:
        return "Please enter a valid client name or ensure cases are loaded.", "", "", ""

    client_names = [case.get("client", "") for case in cases]
    client_embeddings = embedder.encode(client_names)
    query_embedding = embedder.encode([client_name])
    semantic_scores = cosine_similarity(query_embedding, client_embeddings)[0]
    best_sem_index = np.argmax(semantic_scores)
    semantic_score = semantic_scores[best_sem_index]
    fuzzy_scores = [fuzz.partial_ratio(client_name.lower(), name.lower()) for name in client_names]
    best_fuzzy_index = np.argmax(fuzzy_scores)
    fuzzy_score = fuzzy_scores[best_fuzzy_index]

    final_index = best_sem_index if semantic_score > 0.6 else best_fuzzy_index if fuzzy_score > 80 else None

    if final_index is None:
        return "No matching case found.", "", "", ""

    case = cases[final_index]
    name = case.get("client", "N/A")
    charges = ", ".join(case.get("charges", []))
    summary = case.get("summaries", "No summary available.")

    try:
        simplified = gemini.generate_content(f"Summarize this case in simple terms:\n\n{summary}").text
        suggestions = gemini.generate_content(f"You are a legal expert. Based on this case, give next legal steps, key issues, and suggestions.\n\n{summary}").text
        answer = ""
        if question:
            qa_prompt = f"Based on this case:\n\n{summary}\n\nAnswer this question:\n{question}"
            answer = gemini.generate_content(qa_prompt).text
    except Exception as e:
        simplified = f"Gemini Error: {e}"
        suggestions = "Gemini couldn't provide suggestions."
        answer = "Gemini couldn't answer the question."

    output_summary = f"**Name**: {name}\n\n**Charges**: {charges}\n\n**Summary**: {summary}"
    return output_summary, simplified, suggestions, answer

#query to case match
def query_to_case_match(query, top_k=3):
    if not query:
        return "Please enter a legal query.", "", "", ""

    cases = load_cases()
    if not cases:
        return "No cases available.", "", "", ""

    combined_summaries = [f"{case.get('client', '')}: {case['summaries']}" for case in cases]
    summary_embeddings = embedder.encode(combined_summaries)
    query_embedding = embedder.encode([query])
    similarity_scores = cosine_similarity(query_embedding, summary_embeddings)[0]

    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    results = []
    full_text = ""
    for idx in top_indices:
        case = cases[idx]
        name = case.get("client", "N/A")
        charges = ", ".join(case.get("charges", []))
        summary = case.get("summaries", "No summary available.")
        try:
            simplified = gemini.generate_content(f"Summarize this case in simple terms:\n\n{summary}").text
        except:
            simplified = "Could not generate simplified summary."

        results.append({"name": name, "charges": charges, "summary": summary, "simplified": simplified})
        full_text += f"### 🔹 Case: {name}\n**Charges**: {charges}\n\n**Summary**:\n{simplified}\n\n---\n"

    try:
        combined_text = "\n\n".join([r["summary"] for r in results])
        qa_prompt = f"Based on the following legal cases:\n\n{combined_text}\n\nAnswer this query:\n{query}"
        answer = gemini.generate_content(qa_prompt).text
    except:
        answer = "Could not generate an answer using Gemini."

    return full_text, "", "", answer

#text to speech
def generate_tts(text):
    from gtts import gTTS
    from tempfile import NamedTemporaryFile
    if not text.strip():
        return None
    tts = gTTS(text=text, lang='en')
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

with gr.Blocks(css="lawfirm.css") as iface:
    gr.Markdown("# ⚖️ Law Firm Case Assistant", elem_id="header")
    gr.Markdown("Choose how you want to search legal cases:")

    search_mode = gr.Radio(["Search by Client Name", "Search by Legal Query"], value="Search by Client Name", label="Choose Mode")

    client_input = gr.Textbox(label="Enter Client Name", placeholder="e.g., Meena", visible=True)
    question_input = gr.Textbox(label="Ask a Question", placeholder="e.g., What should I do next?", visible=True)
    query_input = gr.Textbox(label="Enter Legal Query", placeholder="e.g., Who got life imprisonment for dowry?", visible=False)

    search_mode.change(
        fn=lambda mode: (
            gr.update(visible=(mode == "Search by Client Name")),
            gr.update(visible=(mode == "Search by Client Name")),
            gr.update(visible=(mode == "Search by Legal Query"))
        ),
        inputs=search_mode,
        outputs=[client_input, question_input, query_input]
    )

    submit_btn = gr.Button("Search")

    case_output = gr.Markdown()
    simplified_output = gr.Markdown()
    suggestions_output = gr.Markdown()
    answer_output = gr.Textbox(visible=False)
    audio_output = gr.Audio(label="Hear the Answer", type="filepath", autoplay=True)

    def main_dispatch(mode, client, ques, query):
        if mode == "Search by Client Name":
            return case_assistant(client, ques)
        elif mode == "Search by Legal Query":
            return query_to_case_match(query)

    submit_btn.click(
        fn=main_dispatch,
        inputs=[search_mode, client_input, question_input, query_input],
        outputs=[case_output, simplified_output, suggestions_output, answer_output]
    ).then(
        fn=generate_tts,
        inputs=answer_output,
        outputs=audio_output,
        preprocess=False
    )

iface.launch()

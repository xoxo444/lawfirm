import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from dotenv import load_dotenv
from docx import Document
import google.generativeai as genai
import numpy as np
import os

custom_css = """
body, .gradio-container {
  background-color: black;
  color: #00ffff !important;
}

.dark body, .dark .gradio-container {
  background-color: black !important;
  color: #fff !important;
}


#header h1 {
  color: beige;
  font-size: 36px;
  text-align: center;
  margin-bottom: 10px;
}

button {
  background-color: beige !important;
  color: black !important;
  font-weight: bold;
  padding: 10px 20px;
  border-radius: 8px !important;
  border: none;
  font-size: 16px;
}


textarea, input[type="text"] {
  color: white; 
  border: 1px solid #ccc;
  border-radius: 8px;
  font-size: 16px;
  padding: 10px;
  font-family: 'Segoe UI', sans-serif;
}

.gr-markdown-output {
  background-color: black;
  padding: 20px;
  border-radius: 10px;
  font-size: 15px;
  border: 1px solid #ddd;
  line-height: 1.6;
  font-family: 'Georgia', serif;
}
"""

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
gemini = genai.GenerativeModel("gemini-1.5-flash")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_cases(folder_path="C:/Users/CS Tiwari/OneDrive/Desktop/Jiya Tiwari/python.py/reumes"):
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

    print(f"✅ Loaded {len(cases)} cases from {folder_path}")
    return cases

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
        prompt = f"Summarize this case in simple terms:\n\n{summary}"
        simplified = gemini.generate_content(prompt).text

        advice_prompt = f"You are a legal expert. Based on this case, give next legal steps, key issues, and suggestions.\n\n{summary}"
        suggestions = gemini.generate_content(advice_prompt).text

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

# Gradio UI
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("# ⚖️ Law Firm Case Assistant", elem_id="header")
    gr.Markdown("Search legal cases by client name and get summaries, suggestions, and answers using Gemini AI.")

    with gr.Row():
        client_input = gr.Textbox(label="Enter Client Name", placeholder="e.g., Meena")
        question_input = gr.Textbox(label="Ask a Question (Optional)", placeholder="e.g., What should I do next?")

    submit_btn = gr.Button("Search")

    case_output = gr.Markdown()
    simplified_output = gr.Markdown()
    suggestions_output = gr.Markdown()
    answer_output = gr.Markdown()

    submit_btn.click(
        fn=case_assistant,
        inputs=[client_input, question_input],
        outputs=[case_output, simplified_output, suggestions_output, answer_output]
    )

iface.launch()

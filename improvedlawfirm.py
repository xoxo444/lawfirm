import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from dotenv import load_dotenv
import docx2txt
import google.generativeai as genai
import numpy as np
import os

# Load environment variables from .env file (must include GEMINI_API_KEY)
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if api_key:
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini = None
    print("❌ GEMINI_API_KEY not found in .env file.")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to load cases from uploaded DOCX
def load_cases_from_file(file_path):
    if not file_path or not os.path.exists(file_path):
        return [], "⚠️ File not found or not uploaded."

    text = docx2txt.process(file_path)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    cases = []
    current_case = {}

    for line in lines:
        if " v. " in line:
            if current_case:
                cases.append(current_case)
                current_case = {}
            current_case["client"] = line

        elif "summary" in line.lower():
            current_case["summaries"] = line.split(":", 1)[-1].strip()

        elif "section" in line.lower():
            charges = line.split(":", 1)[-1].strip()
            current_case["charges"] = [s.strip() for s in charges.split(",")]

    if current_case:
        cases.append(current_case)

    return cases, f"✅ Loaded {len(cases)} cases."

# Main logic
def case_assistant(file, client_name, question=""):
    cases, status = load_cases_from_file(file.name if file else None)
    if not client_name or not cases:
        return status, "", "", ""

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
        return "❌ No matching case found.", "", "", ""

    case = cases[final_index]
    name = case.get("client", "N/A")
    charges = ", ".join(case.get("charges", []))
    summary = case.get("summaries", "No summary available.")

    # Gemini-based responses
    if not gemini:
        return "Gemini API key not found.", summary, "Gemini not available.", "Gemini not available."

    try:
        simplified = gemini.generate_content(
            f"Summarize this case in simple terms:\n\n{summary}"
        ).text

        suggestions = gemini.generate_content(
            f"You are a legal expert. Based on this case, give next legal steps, key issues, and suggestions:\n\n{summary}"
        ).text

        answer = ""
        if question:
            answer = gemini.generate_content(
                f"Based on this case:\n\n{summary}\n\nAnswer this question:\n{question}"
            ).text

    except Exception as e:
        simplified = f"Gemini Error: {e}"
        suggestions = "Gemini couldn't provide suggestions."
        answer = "Gemini couldn't answer the question."

    output_summary = f"**Name**: {name}\n\n**Charges**: {charges}\n\n**Summary**: {summary}"
    return output_summary, simplified, suggestions, answer

# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown("## ⚖️ Law Firm Case Assistant")
    gr.Markdown("Upload your case DOCX, enter a client name, and get insights using Gemini AI.")

    file_input = gr.File(label="Upload DOCX Case File (.docx)", file_types=[".docx"])
    client_input = gr.Textbox(label="Enter Client Name", placeholder="e.g., Meena")
    question_input = gr.Textbox(label="Ask a Question (Optional)", placeholder="e.g., What should I do next?")
    submit_btn = gr.Button("Search")

    case_output = gr.Markdown()
    simplified_output = gr.Markdown()
    suggestions_output = gr.Markdown()
    answer_output = gr.Markdown()

    submit_btn.click(
        fn=case_assistant,
        inputs=[file_input, client_input, question_input],
        outputs=[case_output, simplified_output, suggestions_output, answer_output]
    )

iface.launch()

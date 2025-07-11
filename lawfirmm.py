from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import numpy as np
from docx import Document
from dotenv import load_dotenv
import google.generativeai as genai
import os

# Load environment variables from .env (must include GEMINI_API_KEY)
load_dotenv(".env")
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("API Key Loaded:", "Yes" if api_key else "No")
print("Current Directory:", os.getcwd())

# Load model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to load and parse cases from a DOCX file
def load_cases_from_docx(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    document = Document(file_path)
    cases = []
    current_case = {}

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        if " v. " in text:
            if current_case:
                cases.append(current_case)
                current_case = {}
            current_case["client"] = text.strip()

        elif "facts" in text.lower() or "summary" in text.lower():
            if ":" in text:
                current_case["summaries"] = text.split(":", 1)[1].strip()
            else:
                current_case["summaries"] = text.strip()

        elif "section" in text.lower():
            charges = text.split(":", 1)[-1].strip()
            current_case["charges"] = [s.strip() for s in charges.split(",")]

    if current_case:
        cases.append(current_case)

    return cases

# --- USAGE NOTE ---
# Ensure your cases.docx is in the same folder or update the path below
docx_path = "cases.docx"

try:
    cases = load_cases_from_docx(docx_path)
except FileNotFoundError as e:
    print(e)
    exit()

client_names = [case.get("client", "") for case in cases]
client_embeddings = model.encode(client_names)

client_name = input("\nEnter the client's name: ").strip()
if not client_name:
    print("No input provided.")
    exit()

# Semantic and Fuzzy matching
query_embedding = model.encode([client_name])
semantic_scores = cosine_similarity(query_embedding, client_embeddings)[0]
best_sem_index = np.argmax(semantic_scores)
semantic_score = semantic_scores[best_sem_index]

fuzzy_scores = [fuzz.partial_ratio(client_name.lower(), name.lower()) for name in client_names]
best_fuzzy_index = np.argmax(fuzzy_scores)
fuzzy_score = fuzzy_scores[best_fuzzy_index]

# Match logic
if semantic_score > 0.6:
    final_index = best_sem_index
elif fuzzy_score > 80:
    final_index = best_fuzzy_index
else:
    final_index = None

# Output & Gemini Summary
if final_index is not None:
    case = cases[final_index]
    print("\nClosest Match Found:")
    print("Name     :", case.get("client", "N/A"))
    print("Charges  :", ", ".join(case.get("charges", [])))
    print("Summary  :", case.get("summaries", "No summary available."))

    if api_key and "summaries" in case and case["summaries"]:
        try:
            print("\nGemini Summary (Simplified):")
            prompt = f"Summarize this case in simple terms:\n\n{case['summaries']}"
            gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini_model.generate_content(prompt)

            if hasattr(response, "text") and response.text:
                print(response.text)
            else:
                print("Gemini didn't return any text. It may be rate limited or empty.")

        except Exception as e:
            print("Gemini Error:", e)
else:
    print("\nNo close match found.")

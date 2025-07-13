⚖️ Legal Case Matcher Based on Client Name with Gemini AI Summarizer

A Python-based tool that matches legal cases by client name using semantic search, and simplifies case summaries using Gemini (Google AI). Case data is sourced from a .docx file.

📂Project Features

🔍 **Client Name Matching** using Sentence Transformers + Fuzzy Matching  
📄 **Reads Case Data** from a Word `.docx` file  
🤖 **Uses Gemini API** to Simplify Legal Summaries  
🔐 **API key securely managed** via `.env` file  

✅Requirements
- **Python 3.8+**
- Install dependencies via:
```bash
pip install -r requirements.txt
````
▶️How to Run

1. **Place your `cases.docx` file** in the same directory as the script.
2. Create a `.env` file in the root directory and add your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```
3. Run the script:

```bash
python lawfirm_case_matcher.py
```
4. Enter the client’s name (full or partial) when prompted.

---

🔍What the Program Does

* Matches the closest case from your `.docx` using:

  *  SentenceTransformer-based **semantic similarity**
  *  **Fuzzy string matching** for partial names

* Displays:

  *  Client Name
  *  Charges
  *  Case Summary

* Sends the case summary to **Gemini** for simplification, and displays a human-readable explanation.



📝DOCX Format Example

```
Neha v. Suresh (2023)
Summary: Dowry harassment post marriage.
Section: IPC §498A, §406

State v. Meena Kapoor (2020)
Facts: Meena allegedly set her in-laws’ house on fire during a domestic dispute.
Section: IPC §436, §498A
```

* Separate each case by a line.
* "Summary" or "Facts" line must include a colon (`:`).
* "Section" entries should be comma-separated.



🧠Gemini Output Example

```
Enter the client's name: neha

Closest Match Found:
Name     : Neha v. Suresh (2023)
Charges  : IPC §498A, §406 (Criminal breach of trust)
Summary  : Dowry harassment post marriage.

Gemini Summary (Simplified):
Neha reported harassment from her in-laws over dowry after marriage. Legal charges were filed under IPC sections §498A and §406.
```

 Notes
* **Ensure your Gemini API key** is correct and active (free-tier limits apply).
* If you're rate-limited or over quota, the Gemini output may be blank or show an error.



 License
This project is free and open-source for learning and legal research use cases.

 Credits
* Google AI [Gemini API](https://ai.google.dev)
* [Sentence Transformers](https://www.sbert.net/)
* [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)
* [python-docx](https://python-docx.readthedocs.io/)


👩‍💻 Built by: [Jiya Tiwari](https://github.com/xoxo444)







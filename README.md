# lawfirm
A client name-based legal case matcher that uses semantic search and GPT to retrieve and simplify case summaries from a DOCX database.


📂 Project Features
- 🔍 Client Name Matching using Sentence Transformers + Fuzzy Matching  
- 📄 Reads Case Data from a Word (.docx) File
- 🧠 Uses GPT (via OpenAI API) to Simplify Legal Summaries 
- 🔐 API key managed securely via `.env`
  
  Requirements
- Python 3.8+
- Dependencies in `requirements.txt`:
  ```bash
  pip install -r requirements.txt

 How to Run
```bash
python lawfirm_case_matcher.py
```
1. Make sure your `cases.docx` file is in the same directory as the script.
2. Ensure you have a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
3. Run the script using the command above.
4. Enter a client’s name (even partial) when prompted.
5. The program will:
   * Match the closest case using semantic + fuzzy matching
   * Display the charges and case summary (if available)
   * Use GPT to generate a simplified summary (if API key is valid)




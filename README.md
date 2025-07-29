# 📄 Enhanced Persona-Driven Document Intelligence (Round 1B)

A scalable and memory-efficient Python tool that intelligently analyzes multiple PDF documents to extract the most relevant sections based on a **persona** and a specific **job to be done**.

This solution is crafted for the **Adobe India Hackathon 2025 (Round 1B)** and builds upon the foundation of Round 1A, enhanced to support **batch PDF processing via a JSON configuration**. It is designed to extract **actionable insights** for varied use cases such as financial audits, technical research, and enterprise documentation analysis.

---

## ✨ Features

* ✅ **JSON-based Input**: Flexible configuration for multiple PDFs, persona, and job/task.
* 📚 **Batch Processing**: Handles multiple documents in a single run.
* 🧠 **Persona-Aware Analysis**: Uses TF-IDF + Cosine Similarity to rank sections by relevance.
* 🧵 **Memory-Efficient Chunking**: Splits PDFs into page chunks to optimize memory usage.
* 📊 **Dynamic Section Detection**: Detects logical sections based on font sizes and text patterns.
* 🧹 **Text Preprocessing**: Cleans and normalizes content using stopword removal and stemming.
* 📈 **Relevance Ranking**: Scores and ranks document sections based on semantic similarity.
* 📝 **Summarization**: Extractive summarization for high-priority content.
* 🖥️ **CLI Support**: Easy-to-use command-line interface with custom arguments.

---

## 🧠 How It Works

1. **Input Parsing**: Loads persona, job task, and list of PDFs from a JSON file.
2. **PDF Sectioning**: Detects headings/sections using heuristics (font size, bold text).
3. **Text Preprocessing**: Applies stopword removal, tokenization, and stemming.
4. **Relevance Calculation**: Uses TF-IDF and cosine similarity to match sections with persona query.
5. **Ranking & Summarization**: Outputs top-k relevant sections with brief summaries.
6. **Output Generation**: Results saved as structured JSON with extracted content, ranks, and metadata.

---

## 📦 Folder Structure (Example)

```
.
├── main.py
├── Dockerfile
├── input.json
├── challenge1b_output.json
├── /output
│   └── results.json
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

> 💡 Requires **Python 3.6+**

1. **Clone the repository**:

   ```bash
   git clone <your-repo-url>
   cd <your-project-folder>
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources**:

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## 📄 `requirements.txt`

```
PyMuPDF
scikit-learn
nltk
numpy
```

---

## 📝 Sample `input.json`

```json
{
  "documents": [
    { "filename": "document1.pdf" },
    { "filename": "annual_report_2023.pdf" },
    { "filename": "technical_manual.pdf" }
  ],
  "persona": {
    "role": "Financial Analyst"
  },
  "job_to_be_done": {
    "task": "Identify key financial risks and growth opportunities in the annual report"
  }
}
```

---

## 🚀 Usage

> You can run the script directly from the command line.

### 🔹 Basic Syntax:

```bash
python main.py --input input.json --pdf_dir data/ --output results.json --top_k 20
```

### 🔹 Parameters:

| Argument       | Description                                     | Default          |
| -------------- | ----------------------------------------------- | ---------------- |
| `--input`      | Path to your JSON configuration file            | **Required**     |
| `--pdf_dir`    | Folder containing your PDF files                | Same as JSON dir |
| `--output`     | Path to the output JSON file                    | `output.json`    |
| `--top_k`      | Number of top sections to extract and summarize | `15`             |
| `--chunk_size` | Number of pages per memory chunk                | `25`             |

---

## 📤 Sample Output (`results.json`)

```json
{
  "metadata": {
    "input_documents": [
      "annual_report_2023.pdf",
      "document1.pdf",
      "technical_manual.pdf"
    ],
    "persona": "Financial Analyst",
    "job_to_be_done": "Identify key financial risks and growth opportunities in the annual report",
    "processing_timestamp": "2025-07-29T10:30:00.123456"
  },
  "extracted_sections": [
    {
      "document": "annual_report_2023.pdf",
      "section_title": "Risk Factors",
      "importance_rank": 1,
      "page_number": 15
    },
    {
      "document": "annual_report_2023.pdf",
      "section_title": "Future Growth Strategy",
      "importance_rank": 2,
      "page_number": 25
    }
  ],
  "subsection_analysis": [
    {
      "document": "annual_report_2023.pdf",
      "refined_text": "The primary financial risks include market volatility and changes in interest rates. Our exposure to foreign currency fluctuations is another significant concern...",
      "page_number": 15
    },
    {
      "document": "annual_report_2023.pdf",
      "refined_text": "Our growth strategy focuses on expanding into new emerging markets. We plan to invest heavily in research and development to drive innovation...",
      "page_number": 25
    }
  ]
}
```

---

## 🧪 Testing Locally

You can test with your own PDFs by:

1. Placing them in the `data/` folder.
2. Updating the `input.json` file.
3. Running the script with `python main.py ...`

Make sure paths are correct relative to your working directory.

---

## 🧠 Potential Use Cases

* 🔍 Annual Report Analysis
* 📊 Competitive Landscape Mapping
* 📄 Technical Documentation Review
* 📈 Investment Risk Profiling
* 🧑‍⚕️ Clinical Report Filtering
* 🧠 Legal Contract Summarization

---

## 🛡️ License

This project is licensed under the [MIT License](./LICENSE).

---

## 💡 Credits

Crafted for the **Adobe India Hackathon 2025 - Round 1B**
Built with ❤️ using Python, NLP, and PDF intelligence.

---

If you’d like, I can now:

* Export this as a `README.md` file for you
* Help you generate a `requirements.txt` or test your `main.py` for edge cases
* Write your submission paragraph for Unstop or Adobe Hackathon portal

Just let me know!

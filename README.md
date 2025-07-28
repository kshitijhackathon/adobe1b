Of course, here is a comprehensive README file for your GitHub project, written in English.

Enhanced Persona-Driven Document Intelligence (Round 1B)

This project provides an advanced Python script that analyzes multiple PDF documents to identify the most relevant sections based on a specific "persona" and "job to be done." It is designed to accept input from a JSON file, which specifies the list of documents and the persona information required for the analysis.

The tool leverages memory-efficient techniques, making it capable of handling large files and complex documents effectively.

✨ Features

JSON Input: Uses a JSON file for configuration, including the list of PDF files, persona, and job details.

Batch Processing: Processes multiple PDF documents in a single run.

Persona-Based Analysis: Utilizes TF-IDF and Cosine Similarity to evaluate text relevance against a given persona and task.

Memory Efficient: Employs generators and chunking to handle large PDFs, minimizing memory usage and enabling garbage collection.

Advanced Section Detection: Robustly identifies document sections based on headings, font sizes, and text patterns.

Text Preprocessing: Cleans text by removing stopwords, performing stemming, and normalizing content to improve accuracy.

Relevance Ranking: Ranks each section according to its relevance to the specified persona.

Summarization: Generates concise summaries by extracting key sentences from the most important sections.

Command-Line Interface: Supports command-line arguments for ease of use and automation.

⚙️ How It Works

Load Input: The script parses a JSON file to get the list of PDF files, the persona, and the "job to be done."

Extract Sections: It processes each PDF in a memory-efficient manner, identifying chapters or sections based on text size, style, and structure.

Preprocess Text: The text from each section is cleaned—stopwords are removed, and words are reduced to their root form (stemming).

Calculate Relevance: A query is constructed from the persona and job description. Using TfidfVectorizer and cosine_similarity, the script calculates how similar each section is to this query.

Rank and Summarize: Sections are ranked based on their relevance scores. The top-ranked sections are further analyzed to create short, readable summaries.

Generate Output: The final results are saved to a JSON file, detailing the top extracted sections and their corresponding summaries.

📦 Requirements

You will need Python 3.6+ to run this script. The following libraries are required:

PyMuPDF

scikit-learn

nltk

numpy

🚀 Installation

Clone this repository:

Generated bash
git clone <your-repository-url>
cd <your-repository-directory>


Install the required libraries:
You can install the necessary packages using pip. For convenience, you can create a requirements.txt file.

requirements.txt:

Generated code
PyMuPDF
scikit-learn
nltk
numpy
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Then, run the command:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Download NLTK Data:
The first time you run the script, it will automatically attempt to download the punkt (for sentence tokenization) and stopwords corpora. If this fails, you can download them manually by running the following commands in a Python interpreter:

Generated python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
📝 Usage

The script is run from the command line. You must provide an input JSON file.

Step 1: Create an Input JSON File

Create a JSON file (e.g., input.json) that details the documents, persona, and job to be done.

Example input.json:

Generated json
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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
Step 2: Organize Your Files

Place your PDF files (document1.pdf, annual_report_2023.pdf, etc.) into a folder.

Place your input.json file in the main project directory.

Step 3: Run the Script

Execute the main.py script from your command line.

Syntax:

Generated bash
python main.py --input <path_to_json> --pdf_dir <path_to_pdfs> --output <output_file_name> --top_k <number_of_sections>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Example:

Assuming your PDFs are in a folder named data/:

Generated bash
python main.py --input input.json --pdf_dir data/ --output results.json --top_k 20
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

--input: Path to your input JSON file (required).

--pdf_dir: The directory where your PDF files are stored. If not provided, it defaults to the same directory as the input JSON file.

--output: The name of the output JSON file (default: output.json).

--top_k: The number of top relevant sections to analyze (default: 15).

--chunk_size: The number of pages per chunk for memory management (default: 25).

📤 Output Format

The script generates a JSON file containing the results of the analysis.

Example results.json:

Generated json
{
  "metadata": {
    "input_documents": [
      "annual_report_2023.pdf",
      "document1.pdf",
      "technical_manual.pdf"
    ],
    "persona": "Financial Analyst",
    "job_to_be_done": "Identify key financial risks and growth opportunities in the annual report",
    "processing_timestamp": "2024-05-21T10:30:00.123456"
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
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

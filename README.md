# My First RAG App

This is a **Retrieval-Augmented Generation (RAG) app** using a local Qwen 2.5 model. You can upload PDFs, build a vector index with FAISS, and ask questions about your documents using a local LLM.

---

## Features

* Upload PDFs via a Streamlit interface.
* Chunk and embed text using Sentence-Transformers.
* Build a FAISS index for fast retrieval.
* Ask questions and get answers from your documents.
* Fully local; uses Ollama to run Qwen 2.5:1.5B model.

---

## Prerequisites

* Python 3.10+
* [Streamlit](https://streamlit.io/)
* [Ollama](https://ollama.com/) installed and running locally
* Git (optional, for cloning the repo)

---

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Hadiya-Shahid/My-first-rag-app.git
cd My-first-rag-app
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Pull the Qwen 2.5 model via Ollama:

```bash
ollama pull qwen2.5:1.5b
```

## Running the App

Start the Streamlit interface:

```bash
streamlit run app.py
```

* Open your browser at `http://localhost:8501`.
* Upload PDFs in the sidebar and click **Ingest PDFs**.
* Ask questions about your documents in the main panel.

## Notes / Tips

* The app works best with **text-based PDFs**. Scanned images may not be readable.
* Your FAISS index is stored locally in `store/`.
* Keep `store/` and `data/` in `.gitignore` if sharing the repo publicly to avoid large files.
* Adjust chunk size and top-K retrieval in the sidebar for better performance.

This project is free to use and modify.

Made with  by **Hadiya Shahid**
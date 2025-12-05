import streamlit as st
from rag_engine import rag_answer, STORE_PATH
from pathlib import Path
import os
from ingest import ingest

#Streamlit page setup
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) Q &A")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
VECTOR_DIR = Path("STORE_PATH")
VECTOR_DIR.mkdir(exist_ok=True)

#File upload
with st.sidebar:
    st.header("Upload PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        pdf_path = DATA_DIR / uploaded_file.name
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded {uploaded_file.name}")

        if st.button("Ingest Document"):
            if not uploaded_file:
                st.warning("Please upload a PDF file first.")
            else:
                with st.spinner("Ingesting document..."):
                    ingest(str(pdf_path), output_dir=str(VECTOR_DIR))
                st.success("Ingestion complete!")
st.markdown("---")
st.markdown("**Local LLM")
st.write("Ensure Ollama server is running locally.")
st.write("Command: `ollama serve`")

#User query input
st.header("Ask a Question")
question = st.text_input("Enter your question here:")
k = st.slider("Number of relevant chunks to retrieve (k):", min_value=1, max_value=10, value=3)

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                answer = rag_answer(question, store_path=str(VECTOR_DIR))
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                    

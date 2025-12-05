import os
import pickle
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
import numpy as np 
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import faiss


#TEXXT CLEANING
def clean_text(text: str) -> str:
    """Cleans the input text by removing extra whitespace and non-ASCII characters."""
    text = ' '.join(text.split())  # Remove extra whitespace
    text = ''.join(char for char in text if ord(char) < 128)  # Remove non-ASCII characters
    return text

#CHUNKING
def chunk_text(text: str, chunk_size=500, overlap= 100):
    chunks = []
    start = 0
    length = len(text)

    while start<length:
        end = min(start+chunk_size, length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = end - overlap

    return chunks

#pdf loading and text extraction
def load_pdf(pdf_path:str):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    return clean_text(text)

#CREATING EMBEDDING
def embed_chunks(chunks: List[str], model_name= "BAAI/bge-small-en"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings


#BUILDING FAISS INDEX
def build_faiss_index(embeddings: np.ndarray, dimension: int):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

#MAIN INGESTION FUNCTION
def ingest(pdf_path:str, output_dir="store"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading PDF from {pdf_path}...")
    text = load_pdf(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)

    print("Creating embeddings...")
    embeddings = embed_chunks(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings, dimension=embeddings.shape[1])
    
    #saving index and chunks
    pickle.dump(chunks, open(f"{output_dir}/chunks.pkl", "wb"))
    faiss.write_index(index, f"{output_dir}/faiss.index")

    print(f"Ingestion complete. Data saved to {output_dir}.")
 
if __name__ == "__main__":
    ingest(
        r"C:\Users\Administrator\Desktop\RAG\notebook\data\Highlights.pdf",
        output_dir=r"C:\Users\Administrator\Desktop\RAG\notebook\vector_store"
    )

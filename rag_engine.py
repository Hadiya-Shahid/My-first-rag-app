import pickle
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:1.5b"

#loading stored data
from pathlib import Path

STORE_PATH = Path(r"C:\Users\Administrator\Desktop\RAG\notebook\vector_store")

def load_store(store_path = STORE_PATH):
    chunks = pickle.load(open(f"{STORE_PATH}/chunks.pkl", "rb"))
    index = faiss.read_index(f"{STORE_PATH}/faiss.index")
    return chunks, index

#embedding user query
def embed_query(query:str, model_name="BAAI/bge-small-en"):
    model = SentenceTransformer(model_name)
    embedding = model.encode([query], convert_to_numpy=True)
    return embedding

#retrieving relevant chunks
def retrieve(query_embeddings, index, chunks, k=3):
    distances, indices =index.search(query_embeddings, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return "\n".join(retrieved_chunks)

#generating response using Ollama
def generate_answer(context, query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer concisely using provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    response = requests.post(
        OLLAMA_API_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.0
        },
        timeout=120
    )

    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    return answer


#Main RAG function
def rag_answer(query:str, store_path= STORE_PATH):
    print("Loading stored data...")
    chunks, index = load_store(store_path)

    print("Embedding query...")
    query_embedding = embed_query(query)

    print("Retrieving relevant chunks...")
    context = retrieve(query_embedding, index, chunks)

    print("Generating answer...")
    answer = generate_answer(context, query)

    return answer

#Demo
if __name__ == "__main__":
    user_query = "What is the main topic of the document?"
    answer = rag_answer(user_query, store_path="C:\\Users\\Administrator\\Desktop\\RAG\\notebook\\vector_store")
    print("Answer:")
    print(answer)



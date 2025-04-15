import fitz  # PyMuPDF

print(fitz.__version__)

import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb
import openai 
import streamlit as st
import tempfile
import os

# Initialize Embedding Model and Vector Database
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_search")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Chunk text for embeddings
def chunk_text(text, chunk_size=512):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = [tokens[i: i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]

# Store chunks in vector database
def store_chunks(chunks):
    embeddings = model.encode(chunks)
    for i, chunk in enumerate(chunks):
        collection.add(ids=[str(i)], embeddings=[embeddings[i]], metadatas=[{"text": chunk}])

# Query database and generate response
def query_pdf(question):
    query_embedding = model.encode([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    retrieved_text = " ".join([doc["text"] for doc in results["metadatas"][0]])
    
    client = OpenAI(api_key="YOUR_OPENAI_API_KEY")
    response = client.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Use the retrieved context to answer."},
                  {"role": "user", "content": f"Context: {retrieved_text}\n\nQuestion: {question}"}]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI
st.title("PDF Search Utility using RAG")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name
    
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    store_chunks(chunks)
    st.success("PDF processed successfully!")
    os.remove(pdf_path)

query = st.text_input("Ask a question:")
if query:
    answer = query_pdf(query)
    st.write("**Answer:**", answer)

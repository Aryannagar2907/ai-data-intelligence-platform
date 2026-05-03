from openai import OpenAI
import os
import faiss
import numpy as np
from pypdf import PdfReader
import pickle

client = OpenAI()

# ---- PATH CONFIG ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

PDF_PATH = os.path.join(DATA_PATH, "GenAI_RAG_Project_20250721_040404.pdf")
INDEX_PATH = os.path.join(DATA_PATH, "faiss_index.bin")
CHUNK_PATH = os.path.join(DATA_PATH, "chunks.pkl")


# ---- STEP 1: LOAD PDF ----
def load_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    return text


# ---- STEP 2: CHUNK TEXT (with overlap) ----
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ---- STEP 3: EMBEDDINGS ----
def get_embeddings(texts):
    embeddings = []

    for t in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        embeddings.append(response.data[0].embedding)

    return np.array(embeddings).astype("float32")


# ---- BUILD + SAVE INDEX ----
def build_and_save_index():
    print("🔄 Building index...")

    text = load_pdf(PDF_PATH)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNK_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Index saved")


# ---- LOAD INDEX ----
def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNK_PATH):
        build_and_save_index()

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNK_PATH, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


# ---- RETRIEVE ----
def retrieve(query, index, chunks, top_k=3):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    q_embed = np.array([response.data[0].embedding]).astype("float32")

    distances, indices = index.search(q_embed, top_k)

    results = [chunks[i] for i in indices[0]]

    return results


# ---- MAIN RAG ----
def run_rag_agent(question: str):
    index, chunks = load_index()

    context_chunks = retrieve(question, index, chunks)
    context = "\n".join(context_chunks)

    prompt = f"""
You are a helpful assistant.

Answer ONLY from the context below.
If answer is not present, say "I do not know".

Context:
{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()
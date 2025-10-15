import os
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import fitz
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai



app = FastAPI(title="Document Search & RAG API (Gemini)")



load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")


genai.configure(api_key=GEMINI_API_KEY)



EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)
DIMENSION = 384
INDEX_PATH = "faiss_index.bin"
META_PATH = "metadata.json"


if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    index = faiss.IndexFlatL2(DIMENSION)
    metadata = []



def chunk_text(text: str, max_length: int = 500):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def extract_text_from_pdf(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def save_index_and_metadata():
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)



@app.get("/")
def home():
    return {"message": "RAG Document Search API (Gemini) is running!"}


@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()

    if file.content_type == "application/pdf":
        text = extract_text_from_pdf(contents)
    elif file.content_type.startswith("text/"):
        text = contents.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text found in document")

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    index.add(embeddings)

    for chunk in chunks:
        metadata.append({"source_file": file.filename, "chunk": chunk})

    save_index_and_metadata()
    return {"message": f"Indexed {len(chunks)} chunks from {file.filename}"}


class Query(BaseModel):
    query: str

@app.post("/search/")
def search(query: Query):
    if not metadata:
        raise HTTPException(status_code=400, detail="No documents indexed yet")

    q_emb = embedder.encode([query.query], convert_to_numpy=True)
    D, I = index.search(q_emb, k=5)
    retrieved_chunks = [metadata[i]["chunk"] for i in I[0] if 0 <= i < len(metadata)]

    if not retrieved_chunks:
        raise HTTPException(status_code=404, detail="No matching chunks found")

    context = "\n".join(retrieved_chunks)
    prompt = (
        f"Use the following context to answer the user's question clearly and accurately.\n\n"
        f"Context:\n{context}\n\nQuestion: {query.query}\n\nAnswer:"
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")  # Updated model name
        response = model.generate_content(
    "Use the following context to answer the question:\n\n" + prompt
)

        answer = response.text
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

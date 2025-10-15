# Knowledge-Base-Search-Engine
# Document Search & RAG API using Gemini AI

This repository contains a FastAPI backend for a document knowledge-base search engine powered by vector embeddings and Google's Gemini generative AI models.

## Features

- Upload and index documents (PDF and text)
- Use SentenceTransformer embeddings for semantic search
- Store embeddings efficiently with FAISS
- Query documents and get AI-generated answers grounded on relevant content
- Integration with Gemini AI (Google Generative AI) for Retrieval-Augmented Generation (RAG)
- Persistent index and metadata storage for efficient reuse

## Getting Started

### Prerequisites

- Python 3.9+
- Install dependencies from `requirements.txt`
- Google Gemini API key with permissions for generative AI models
- `faiss`, `sentence-transformers`, `PyMuPDF`, `fastapi`, `uvicorn`, and `python-dotenv`

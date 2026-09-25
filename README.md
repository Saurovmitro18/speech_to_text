# 🎙️ Multimodal Audio & Document Intelligence Copilot

An enterprise-grade, end-to-end AI copilot that converts spoken audio into timestamped transcripts, indexes segments into a local vector database, and uses Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to allow users to search, summarize, and query meeting or lecture recordings in real time.

---

## 📌 Project Overview

This project provides an intelligent workflow for audio processing:
1. **Speech Recognition & Diarization**: Converts audio into text using `faster-whisper` and generates precise segment timestamps and speaker metadata.
2. **Vector Indexing & RAG**: Chunks and embeds transcribed audio segments using HuggingFace sentence transformers (`all-MiniLM-L6-v2`) and stores them locally in `ChromaDB`.
3. **RESTful Microservice Backend**: A `FastAPI` service handling asynchronous audio ingestion, vector search, and LLM orchestration.
4. **LLM Synthesis**: Generates structured summaries, action items, and context-aware Q&A using Groq API or local Ollama LLMs.
5. **Interactive Dashboard**: A `Streamlit` user interface for managing uploads, searching transcripts semantically, and viewing structured summaries.

---

## 🛠️ Tech Stack & Architecture

- **Language**: Python 3.11.8
- **Audio Processing & ASR**: `faster-whisper`, `FFmpeg`, `pyannote.audio`
- **Embeddings & Vector Database**: `langchain-huggingface`, `sentence-transformers`, `ChromaDB`
- **Backend API**: `FastAPI`, `Uvicorn`
- **LLM Providers**: `Groq API` / `Ollama`
- **Frontend UI**: `Streamlit` (Track 1) / `Gradio` (Simple STT Utility)

---

## 📁 Repository Structure

```text
speech_to_text_app/
│
├── chroma_db/               # Local Chroma Vector Database store
├── venv/                    # Local Python virtual environment
│
├── app.py                   # Legacy / Standalone Gradio STT & Synonym utility
├── diarization_pipeline.py  # Audio ASR engine (faster-whisper + timestamps)
├── rag_engine.py            # Vector embedding and ChromaDB retrieval pipeline
├── main.py                  # FastAPI REST backend endpoints
├── app_streamlit.py         # Main Streamlit interactive copilot UI
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

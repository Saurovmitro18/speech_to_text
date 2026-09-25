import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Import processing functions from Step 2 & Step 3
from diarization_pipeline import process_audio_with_timestamps
from rag_engine import build_vector_store, search_transcript

app = FastAPI(title="Multimodal Meeting Intelligence API", version="1.0")

# Global in-memory reference to store current active vector DB instance
ACTIVE_VECTORSTORE = None
TRANSCRIPT_CACHE = []

class QueryRequest(BaseModel):
    query: str

class SummaryRequest(BaseModel):
    llm_provider: str = "groq"  # Choices: "groq" or "ollama"

@app.get("/")
def read_root():
    return {"status": "Online", "service": "Multimodal Meeting Copilot API"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Receives an audio file, processes timestamps with Whisper, 
    and indexes segments into ChromaDB.
    """
    global ACTIVE_VECTORSTORE, TRANSCRIPT_CACHE
    
    # Save uploaded file temporarily to disk
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Extract timestamped transcript blocks
        transcript_blocks = process_audio_with_timestamps(temp_file_path)
        TRANSCRIPT_CACHE = transcript_blocks
        
        # 2. Vectorize and index in ChromaDB
        ACTIVE_VECTORSTORE = build_vector_store(transcript_blocks)
        
        return {
            "status": "Success",
            "filename": file.filename,
            "segments_count": len(transcript_blocks),
            "transcript": transcript_blocks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up local temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/search-transcript/")
def search_meeting(request: QueryRequest):
    """
    Searches the stored meeting vector database for relevant transcript segments.
    """
    global ACTIVE_VECTORSTORE
    if not ACTIVE_VECTORSTORE:
        raise HTTPException(status_code=400, detail="No transcript indexed yet. Upload an audio file first.")
    
    matches = search_transcript(request.query, ACTIVE_VECTORSTORE, top_k=3)
    return {"query": request.query, "results": matches}

@app.post("/generate-summary/")
def generate_summary(request: SummaryRequest):
    """
    Generates key summaries and action items from the meeting transcript using an LLM.
    """
    global TRANSCRIPT_CACHE
    if not TRANSCRIPT_CACHE:
        raise HTTPException(status_code=400, detail="No active meeting transcript found.")

    # Combine full meeting transcript
    full_text = "\n".join([f"[{b['start']}s-{b['end']}s] {b['speaker']}: {b['text']}" for b in TRANSCRIPT_CACHE])
    prompt = f"Analyze this meeting transcript and summarize the key decisions and action items:\n\n{full_text}"

    # Route request based on selected LLM provider
    if request.llm_provider.lower() == "groq":
        try:
            from groq import Groq
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            return {"summary": response.choices[0].message.content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")

    elif request.llm_provider.lower() == "ollama":
        try:
            import ollama
            response = ollama.chat(
                model="llama3.1",
                messages=[{"role": "user", "content": prompt}]
            )
            return {"summary": response.message.content}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ollama Local Error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Unsupported provider. Choose 'groq' or 'ollama'.")
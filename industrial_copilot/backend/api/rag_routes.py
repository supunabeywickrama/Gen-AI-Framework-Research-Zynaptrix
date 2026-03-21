from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import shutil
import os

from rag_ingestion.pipeline import process_manual
from rag_retrieval.rag import RAGGenerator

router = APIRouter()
rag_gen = RAGGenerator()

class ChatRequest(BaseModel):
    manual_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str
    images: list[str]
    pages: list[int]

@router.post("/ingest-manual")
async def ingest_manual(
    background_tasks: BackgroundTasks,
    manual_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{manual_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = process_manual(file_path, manual_id)
        return {"message": "Manual ingested successfully", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_data = rag_gen.generate_response(request.query, request.manual_id)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

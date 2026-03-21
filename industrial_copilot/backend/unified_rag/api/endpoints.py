from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
import shutil
import os
from unified_rag.ingestion.pipeline import process_manual
from unified_rag.retrieval.rag import RAGGenerator

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
    print(f"\n🚀 [API] Received ingestion request for Manual ID: {manual_id}")
    print(f"📄 [API] File: {file.filename} (Size roughly: {file.size if hasattr(file, 'size') else 'unknown'} bytes)")

    if not file.filename.endswith(".pdf"):
        print(f"❌ [API] Rejected: {file.filename} is not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{manual_id}_{file.filename}")
    
    print(f"💾 [API] Saving file to: {file_path}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"✅ [API] File saved. Starting ingestion pipeline...")
        
    # Process immediately
    try:
        result = process_manual(file_path, manual_id)
        print(f"🏁 [API] Ingestion successful for {manual_id}!")
        return {"message": "Manual ingested successfully", "details": result}
    except Exception as e:
        print(f"🔥 [API] CRITICAL ERROR during ingestion: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_data = rag_gen.generate_response(request.query, request.manual_id)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

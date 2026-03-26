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

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine
from sqlalchemy.orm import Session
from fastapi import Depends

# ... (existing imports)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class MachineCreate(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str

class MachineResponse(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str
    class Config:
        from_attributes = True

@router.post("/machines", response_model=MachineResponse)
async def create_machine(machine: MachineCreate, db: Session = Depends(get_db)):
    # Check if machine already exists to avoid 500/IntegrityError
    existing = db.query(Machine).filter(Machine.machine_id == machine.machine_id).first()
    if existing:
        # Update existing record if needed
        for key, value in machine.model_dump().items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine.model_dump())
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.get("/machines", response_model=list[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

@router.post("/machines/delete/{machine_id}")
async def delete_machine(machine_id: str, db: Session = Depends(get_db)):
    print(f"🗑️ [API] Deletion request (POST) for Machine ID: {machine_id}")
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        print(f"⚠️ [API] Machine {machine_id} not found for deletion")
        raise HTTPException(status_code=404, detail="Machine not found")
    
    db.delete(machine)
    db.commit()
    print(f"✅ [API] Machine {machine_id} successfully decommissioned")
    return {"message": f"Machine {machine_id} deleted successfully"}

@router.get("/machines/{machine_id}", response_model=MachineResponse)
async def get_machine(machine_id: str, db: Session = Depends(get_db)):
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response_data = rag_gen.generate_response(request.query, request.manual_id)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

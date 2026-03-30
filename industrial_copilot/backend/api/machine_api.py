from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import Machine, AnomalyRecord, ChatMessage, InteractionMemory
from unified_rag.config import settings
from pydantic import BaseModel
import json
import os
import openai
from datetime import datetime

router = APIRouter(tags=["Machine Registry"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class MachineBase(BaseModel):
    machine_id: str
    name: str
    location: str
    manual_id: str

class MachineResponse(MachineBase):
    class Config:
        from_attributes = True

class AnomalyResponse(BaseModel):
    id: int
    machine_id: str
    timestamp: str
    type: str
    score: int
    sensor_data: str # JSON backend

    class Config:
        from_attributes = True

class ResolveRequest(BaseModel):
    operator_fix: str

@router.get("/api/chat-history/{anomaly_id}")
async def get_chat_history(anomaly_id: int):
    db = SessionLocal()
    try:
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).order_by(ChatMessage.id).all()
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "images": json.loads(m.images) if m.images else []
            } 
            for m in messages
        ]
    finally:
        db.close()

@router.post("/api/chat-history/{anomaly_id}/resolve")
async def resolve_incident(anomaly_id: int, req: ResolveRequest):
    db = SessionLocal()
    try:
        # 1. Mark Anomaly as Resolved
        record = db.query(AnomalyRecord).filter(AnomalyRecord.id == anomaly_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Incident not found")
        record.resolved = True
        
        # 2. Extract History
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).all()
        history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        # 3. Use LLM to summarize Technical Solution (Action-Centric)
        openai.api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        summary_prompt = (
            "You are a Technical Scribe. BELOW IS A DIAGNOSTIC CHAT HISTORY and an actual manual fix.\n"
            "Summarize them into a concise, action-oriented entry (max 150 words) for future retrieval.\n\n"
            f"History:\n{history_text}\n\nOperator Fix: {req.operator_fix}"
        )
        res = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}])
        summary = res.choices[0].message.content
        
        # 4. Fetch Embedding
        emb_res = openai.embeddings.create(input=summary, model="text-embedding-3-small")
        embedding = emb_res.data[0].embedding
        
        # 5. Store in InteractionMemory registry
        mem = InteractionMemory(
            machine_id=record.machine_id,
            manual_id="Historical_Knowledge",
            summary=summary,
            operator_fix=req.operator_fix,
            embedding=embedding,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.add(mem)
        db.commit()
        return {"status": "resolved_and_archived", "summary": summary}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@router.get("/api/machines", response_model=List[MachineResponse])
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()

@router.get("/api/machines/{machine_id}/anomalies", response_model=List[AnomalyResponse])
async def get_machine_anomalies(machine_id: str, db: Session = Depends(get_db)):
    """Fetch history of anomalies for a specific machine."""
    return db.query(AnomalyRecord).filter(AnomalyRecord.machine_id == machine_id).order_by(AnomalyRecord.id.desc()).all()

@router.post("/api/machines", response_model=MachineResponse)
async def register_machine(machine: dict, db: Session = Depends(get_db)):
    """Add or update a machine in the registry."""
    existing = db.query(Machine).filter(Machine.machine_id == machine.get('machine_id')).first()
    if existing:
        for key, value in machine.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine)
    db.add(db_machine)
    db.commit()
    db.refresh(db_machine)
    return db_machine

@router.post("/api/machines/delete/{machine_id}")
async def decommission_machine(machine_id: str, db: Session = Depends(get_db)):
    """Remove a machine from the registry."""
    machine = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if not machine:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    db.delete(machine)
    db.commit()
    return {"status": "decommissioned", "machine_id": machine_id}

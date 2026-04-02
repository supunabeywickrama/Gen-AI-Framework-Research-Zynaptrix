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
from fastapi import Request, UploadFile
import subprocess
import asyncio
from services.datasheet_parser import DatasheetParser

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
    resolved: bool

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
                "images": json.loads(m.images) if m.images else [],
                "metadata": json.loads(m.message_metadata) if m.message_metadata else None,
                "db_id": m.id
            } 
            for m in messages
        ]
    finally:
        db.close()

class TaskUpdateRequest(BaseModel):
    task_id: str
    completed: bool

@router.patch("/api/chat-message/{message_id}/task")
async def update_task_status(message_id: int, req: TaskUpdateRequest):
    """Update a single task completion status within a procedure stored in a chat message."""
    db = SessionLocal()
    try:
        msg = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if not msg:
            raise HTTPException(status_code=404, detail="Message not found")
        
        meta = json.loads(msg.message_metadata) if msg.message_metadata else {}
        if "completed_tasks" not in meta:
            meta["completed_tasks"] = {}
        meta["completed_tasks"][req.task_id] = req.completed
        msg.message_metadata = json.dumps(meta)
        db.commit()
        return {"status": "updated", "completed_tasks": meta["completed_tasks"]}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
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
        
        # 5. Store in InteractionMemory registry (for RAG)
        mem = InteractionMemory(
            machine_id=record.machine_id,
            manual_id="Historical_Knowledge",
            summary=summary,
            operator_fix=req.operator_fix,
            embedding=embedding,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        db.add(mem)

        # 6. Store as a ChatMessage (for word-for-word history & archival view)
        summary_msg = ChatMessage(
            anomaly_id=anomaly_id,
            role='agent',
            content=f"### 🏁 Final Completion Report\n\n**Technical Summary**:\n{summary}\n\n**Operator Notes**:\n{req.operator_fix}",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            message_metadata=json.dumps({"type": "final_summary"})
        )
        db.add(summary_msg)
        
        # 7. Mark as resolved in DB
        record.resolved = True
        
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
async def register_machine(request: Request, db: Session = Depends(get_db)):
    """Add or update a machine in the registry via FormData (supports PDFs)."""
    form_data = await request.form()
    
    machine_id = str(form_data.get('machine_id'))
    name = str(form_data.get('name') or "")
    location = str(form_data.get('location') or "")
    manual_id = str(form_data.get('manual_id') or "")

    if not machine_id or machine_id == "None":
        raise HTTPException(status_code=400, detail="machine_id is required")

    machine_dict = {
        "machine_id": machine_id,
        "name": name,
        "location": location,
        "manual_id": manual_id
    }

    # 1. Parse dynamically passed sensors from form_data
    sensors_json = form_data.get('sensors')
    sensor_configs = {}
    
    if sensors_json:
        try:
            sensors = json.loads(str(sensors_json))
            parser = DatasheetParser(os.getenv("OPENAI_API_KEY", settings.openai_api_key))
            
            for sensor in sensors:
                s_name = sensor.get("sensor_name", "")
                s_id = sensor.get("sensor_id", "")
                file_key = f"datasheet_{s_id}"
                
                pdf_file: UploadFile = form_data.get(file_key)
                if pdf_file and pdf_file.filename:
                    pdf_bytes = await pdf_file.read()
                    pdf_text = parser.parse_pdf(pdf_bytes)
                    config = parser.extract_sensor_config(s_name, s_id, pdf_text)
                else:
                    # No PDF — still use OpenAI to estimate from sensor name
                    config = parser.extract_sensor_config_no_pdf(s_name, s_id)
                sensor_configs[s_id] = config

            # Save the configurations
            config_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sensor_configs.json")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            all_configs = {}
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    all_configs = json.load(f)
            
            all_configs[machine_id] = sensor_configs
            with open(config_path, "w") as f:
                json.dump(all_configs, f, indent=2)
                
            # 2. Trigger async dataset generation and model training
            def run_generation():
                backend_dir = os.path.join(os.path.dirname(__file__), "..")
                try:
                    subprocess.run(["python", "generate_dataset.py", "--machine_id", machine_id], cwd=backend_dir, check=True)
                    # New: Mandatory Normalization Step
                    subprocess.run(["python", "preprocessing/normalization.py", "--machine_id", machine_id], cwd=backend_dir, check=True)
                    subprocess.run(["python", "models/train_model.py", "--machine_id", machine_id], cwd=backend_dir, check=True)
                except Exception as e:
                    print(f"Failed to generate datasets/models for {machine_id}: {e}")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_generation)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse sensors/datasheets: {e}")

    # 3. Update Database
    existing = db.query(Machine).filter(Machine.machine_id == machine_id).first()
    if existing:
        for key, value in machine_dict.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
        
    db_machine = Machine(**machine_dict)
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

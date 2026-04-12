from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
import os
import logging
import openai
from pydantic import BaseModel

from unified_rag.db.database import SessionLocal
from unified_rag.db.models import AnomalyRecord, ChatMessage, InteractionMemory
from unified_rag.config import settings

router = APIRouter(prefix="/api/copilot/chat", tags=["Copilot Chat"])

class ResolveRequest(BaseModel):
    operator_fix: str

@router.get("/{anomaly_id}")
async def get_chat_history(anomaly_id: str):
    print(f"DEBUG: get_chat_history called for anomaly_id={anomaly_id}")
    db = SessionLocal()
    try:
        # Convert to int for query
        try:
            aid = int(anomaly_id)
        except:
            return []
            
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == aid).order_by(ChatMessage.id).all()
        print(f"DEBUG: Found {len(messages)} messages")
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

@router.post("/{anomaly_id}/resolve")
async def resolve_incident(anomaly_id: int, req: ResolveRequest):
    db = SessionLocal()
    try:
        # 1. Mark Anomaly as Resolved
        record = db.query(AnomalyRecord).filter(AnomalyRecord.id == anomaly_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Incident not found")
        record.resolved = True
        
        # 2. Extract Chat History
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).all()
        history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        # 3. Summarize Fix
        openai.api_key = os.getenv("OPENAI_API_KEY", settings.openai_api_key)
        summary_prompt = (
            "You are a Technical Scribe. Summarize the troubleshooting steps and the operator's fix into a concise action entry.\n\n"
            f"History:\n{history_text}\n\nOperator Fix: {req.operator_fix}"
        )
        
        res = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = res.choices[0].message.content
        
        # 4. Vectorize
        emb_res = openai.embeddings.create(input=summary, model="text-embedding-3-small")
        embedding = emb_res.data[0].embedding
        
        # 5. Archive
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
        return {"status": "resolved", "summary": summary}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

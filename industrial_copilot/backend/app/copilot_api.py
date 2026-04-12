import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI

from agents.copilot_graph import build_copilot_graph
from unified_rag.db.database import SessionLocal, get_db
from unified_rag.db.models import AnomalyRecord, ChatMessage, InteractionMemory
from unified_rag.config import settings

router = APIRouter(prefix="/api/copilot", tags=["Diagnostic Copilot"])
logger = logging.getLogger(__name__)

# Initialize the Multi-Agent Flow
copilot_workflow = build_copilot_graph()

class AnomalyEvent(BaseModel):
    machine_id: str = "PUMP-001"
    machine_state: str = "manual_inquiry"
    anomaly_id: Optional[int] = None
    anomaly_score: Optional[float] = 0.0
    user_query: Optional[str] = None
    suspect_sensor: Optional[str] = "Unknown"
    recent_readings: Optional[Dict[str, Any]] = None

class IntentRequest(BaseModel):
    user_message: str
    step_text: str
    machine_id: str = "PUMP-001"

class ResolveRequest(BaseModel):
    operator_fix: str

@router.post("/invoke")
def invoke_copilot(event: AnomalyEvent, db: Session = Depends(get_db)):
    """Entry point for the Multi-Agent Diagnostic Orchestration."""
    
    # 🧠 PHASE 1: Context Hydration
    chat_context = ""
    look_id = int(event.anomaly_id) if event.anomaly_id else None
    
    if look_id:
        past_messages = db.query(ChatMessage).filter(
            ChatMessage.anomaly_id == look_id
        ).order_by(ChatMessage.id).all()
        
        for m in past_messages:
            chat_context += f"{m.role.upper()}: {m.content}\n"

    # 📝 PHASE 2: Persistent User Message
    actual_id = None
    if look_id:
        anomaly_record = db.query(AnomalyRecord).filter(AnomalyRecord.id == look_id).first()
        if anomaly_record:
            actual_id = anomaly_record.id
            
            if event.user_query:
                # Save user message
                metadata = None
                if "[CONVERSATIONAL_WIZARD]" in event.user_query:
                    intent_match = event.user_query.split("(Context: ")
                    if len(intent_match) > 1:
                        intent_label = intent_match[1].replace(")", "")
                        metadata = {"action": "step_response", "status": "done" if "CONFIRM_DONE" in intent_label else "cant_do"}
                
                msg = ChatMessage(
                    anomaly_id=actual_id,
                    role='user',
                    content=event.user_query,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    message_metadata=json.dumps(metadata) if metadata else None
                )
                db.add(msg)
                db.commit()

    initial_state = {
        "event_id": f"EVT-{actual_id}" if actual_id else "EVT-LIVE-QUERY",
        "machine_id": event.machine_id,
        "machine_state": event.machine_state,
        "anomaly_score": event.anomaly_score or 0.0,
        "user_query": event.user_query,
        "chat_history": chat_context,
        "sensor_status_report": "",
        "diagnostic_report": "",
        "rag_context": "",
        "retrieved_images": [],
        "strategy_report": "",
        "critic_feedback": "",
        "final_execution_plan": ""
    }

    # 🚀 PHASE 3: Agent Orchestration
    try:
        result = copilot_workflow.invoke(initial_state)
    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        return {"status": "error", "message": f"Orchestration failure: {str(e)}"}

    # 🛡️ PHASE 4: Persistent Agent Response
    final_answer = result.get("final_execution_plan", "")
    if final_answer and actual_id:
        try:
            msg = ChatMessage(
                anomaly_id=actual_id,
                role='agent',
                content=final_answer,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                images=json.dumps(result.get("retrieved_images", []))
            )
            db.add(msg)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to store agent response: {e}")

    return {"status": "success", "graph_result": result, "stored_id": actual_id}

@router.post("/classify-intent")
async def classify_intent(req: IntentRequest):
    """Determines user intent during a repair procedure (HITL Layer)."""
    openai_client = OpenAI(api_key=settings.openai_api_key)

    system_prompt = (
        "You are an intent classifier for an industrial maintenance assistant. "
        "A technician is working on a repair procedure. "
        "Based on their message and the current step context, classify their intent into EXACTLY ONE of these categories:\n\n"
        "CONFIRM_DONE - They completed the step\n"
        "NEED_HELP - They are stuck or have a problem\n"
        "NEED_DETAIL - They want more details on how to do the step\n"
        "FREE_CHAT - General question not about completing this step\n\n"
        "Reply with ONLY one of these four words."
    )
    user_prompt = f"Current repair step: \"{req.step_text}\"\n\nTechnician message: \"{req.user_message}\""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        intent = response.choices[0].message.content.strip().upper()
        if intent not in {"CONFIRM_DONE", "NEED_HELP", "NEED_DETAIL", "FREE_CHAT"}:
            intent = "FREE_CHAT"
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        intent = "FREE_CHAT"

    return {"intent": intent}

@router.get("/chat/{anomaly_id}")
async def get_chat_history(anomaly_id: int, db: Session = Depends(get_db)):
    """Returns the chat history for a specific anomaly incident."""
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

@router.post("/chat/{anomaly_id}/resolve")
async def resolve_incident(anomaly_id: int, req: ResolveRequest, db: Session = Depends(get_db)):
    """Marks an anomaly as resolved and archives the successful resolution path."""
    try:
        # 1. Mark Anomaly as Resolved
        record = db.query(AnomalyRecord).filter(AnomalyRecord.id == anomaly_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="Incident not found")
        record.resolved = True
        
        # 2. Extract Chat History
        messages = db.query(ChatMessage).filter(ChatMessage.anomaly_id == anomaly_id).all()
        history_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
        
        # 3. AI Summarization
        openai_client = OpenAI(api_key=settings.openai_api_key)
        summary_prompt = (
            "Summarize the troubleshooting steps and the operator's fix into a concise technical action entry.\n\n"
            f"History:\n{history_text}\n\nOperator Fix: {req.operator_fix}"
        )
        
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        summary = res.choices[0].message.content
        
        # 4. Vectorize for Interaction Memory
        emb_res = openai_client.embeddings.create(input=summary, model="text-embedding-3-small")
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
        logger.error(f"Failed to resolve incident: {e}")
        raise HTTPException(status_code=500, detail=str(e))

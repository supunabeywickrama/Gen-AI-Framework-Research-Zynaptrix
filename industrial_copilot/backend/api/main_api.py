import logging
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from agents.copilot_graph import build_copilot_graph
from unified_rag.api.endpoints import router as rag_router
from unified_rag.db.models import Machine, AnomalyRecord
from unified_rag.db.database import engine, Base, SessionLocal
from sqlalchemy import text
from services.anomaly_service import AnomalyService

logging.basicConfig(level=logging.INFO)

try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
except Exception as e:
    logging.warning(f"Vector extension issue: {e}")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Generative AI Multi-Agent Industrial Copilot")

# Ensure the local `data` directory exists for static mounting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
os.makedirs(os.path.join(DATA_DIR, "extracted"), exist_ok=True)

# Mount with Absolute Path for stability across different execution contexts
app.mount("/static", StaticFiles(directory=DATA_DIR), name="static")
logging.info(f"✅ Static assets mounted from: {DATA_DIR}")

app.include_router(rag_router, tags=["Knowledge Base"])

# Ensure Next.js Frontend can securely communicate with Backend
import os
frontend_url = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Multi-Agent Flow
copilot_workflow = build_copilot_graph()

class TelemetryClientManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

telemetry_manager = TelemetryClientManager()

from unified_rag.db.models import Machine, AnomalyRecord, ChatMessage, InteractionMemory

class AnomalyEvent(BaseModel):
    machine_id: str = "PUMP-001"
    machine_state: str = "manual_inquiry"
    anomaly_id: Optional[int] = None # Linked incident ID
    anomaly_score: Optional[float] = 0.0
    user_query: Optional[str] = None
    suspect_sensor: Optional[str] = "Unknown"
    recent_readings: Optional[Dict[str, Any]] = None

@app.post("/api/copilot/invoke")
def invoke_copilot(event: AnomalyEvent):
    """Synchronous entry point to avoid blocking event loop during heavy RAG."""
    with open("api_debug.log", "a") as f:
        f.write(f"\n--- INQUIRY START: Anomaly #{event.anomaly_id} ---\n")
        
        # 📝 PHASE 1: Persistent User Message (Isolated Session)
        actual_id = None
        try:
            db_user = SessionLocal()
            try:
                # Try to map to integer for DB lookup if possible
                look_id = int(event.anomaly_id) if event.anomaly_id else None
                anomaly_record = db_user.query(AnomalyRecord).filter(AnomalyRecord.id == look_id).first()
                if anomaly_record:
                    actual_id = anomaly_record.id
                
                if event.user_query:
                    msg = ChatMessage(
                        anomaly_id=actual_id,
                        role='user',
                        content=event.user_query,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                    db_user.add(msg)
                    db_user.commit()
            except Exception as e:
                f.write(f"DB_WARNING[USER_PHASE]: Connection failed, proceeding anonymously: {e}\n")
            finally:
                db_user.close()
        except Exception as e:
            f.write(f"DB_CRITICAL[INIT]: SessionLocal failed: {e}\n")

        # 🧠 PHASE 2: Context Hydration
        # Fetch the active chat history to provide 'Conversational Awareness' to the Graph
        chat_context = ""
        try:
            db_history = SessionLocal()
            try:
                past_messages = db_history.query(ChatMessage).filter(
                    ChatMessage.anomaly_id == actual_id
                ).order_by(ChatMessage.id).all()
                
                for m in past_messages:
                    chat_context += f"{m.role.upper()}: {m.content}\n"
                
                f.write(f"CONTEXT: Hydrated {len(past_messages)} historical messages for {actual_id}.\n")
            except Exception as e:
                f.write(f"HISTORY_WARNING: DB offline, providing zero-shot context only: {e}\n")
            finally:
                db_history.close()
        except:
            pass

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
            f.write("SUCCESS: Graph execution completed.\n")
        except Exception as e:
            f.write(f"GRAPH_ERROR: {e}\n")
            return {"status": "error", "message": f"Orchestration failure: {str(e)}"}

        # 🛡️ PHASE 4: Persistent Agent Response
        final_answer = result.get("final_execution_plan", "")
        if final_answer:
            try:
                db_agent = SessionLocal()
                try:
                    msg = ChatMessage(
                        anomaly_id=actual_id,
                        role='agent',
                        content=final_answer,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        images=json.dumps(result.get("retrieved_images", []))
                    )
                    db_agent.add(msg)
                    db_agent.commit()
                    f.write(f"SUCCESS: Agent response stored for ID {actual_id}.\n")
                except Exception as e:
                    f.write(f"DB_WARNING[AGENT_PHASE]: Storage failed: {e}\n")
                finally:
                    db_agent.close()
            except:
                pass
                
        return {"status": "success", "graph_result": result, "stored_id": actual_id}

from api.machine_api import router as machine_registry_router
app.include_router(machine_registry_router)

class IntentRequest(BaseModel):
    user_message: str
    step_text: str  # The current repair step text for context
    machine_id: str = "PUMP-001"

@app.post("/api/copilot/classify-intent")
async def classify_intent(req: IntentRequest):
    """
    Fast intent classification using gpt-4o-mini.
    Determines what a user is trying to communicate in the context of a repair step.
    Returns one of: CONFIRM_DONE | NEED_HELP | NEED_DETAIL | FREE_CHAT
    """
    import openai
    from unified_rag.config import settings
    openai.api_key = settings.openai_api_key

    system_prompt = (
        "You are an intent classifier for an industrial maintenance assistant. "
        "A technician is working on a repair procedure. "
        "Based on their message and the current step context, classify their intent into EXACTLY ONE of these categories:\n\n"
        "CONFIRM_DONE - They are saying they completed the step (e.g. 'done', 'finished', 'I tightened the bolt', 'all good', 'checked it')\n"
        "NEED_HELP - They are stuck, have a problem, or something went wrong (e.g. 'can't find', 'broken', 'seized', 'wrong', 'error', 'doesn't work', 'stuck')\n"
        "NEED_DETAIL - They want to understand how to do something (e.g. 'how', 'what is', 'explain', 'show me', 'I don't understand', 'what does X mean')\n"
        "FREE_CHAT - A general question not directly about completing this step (e.g. 'what's the temperature limit', 'how old is this machine')\n\n"
        "Reply with ONLY one of these four words: CONFIRM_DONE, NEED_HELP, NEED_DETAIL, or FREE_CHAT."
    )
    user_prompt = f"Current repair step: \"{req.step_text}\"\n\nTechnician message: \"{req.user_message}\""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=10,
            temperature=0.0  # Deterministic classification
        )
        intent = response.choices[0].message.content.strip().upper()
        # Validate
        valid = {"CONFIRM_DONE", "NEED_HELP", "NEED_DETAIL", "FREE_CHAT"}
        if intent not in valid:
            intent = "FREE_CHAT"
    except Exception as e:
        logging.error(f"Intent classification failed: {e}")
        intent = "FREE_CHAT"

    return {"intent": intent}


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)

# --- Real-Time Agentic Orchestration Layer ---

async def async_broadcast_anomaly(result):
    await telemetry_manager.broadcast(json.dumps({
        "type": "anomaly_alert", 
        "data": result
    }))

def on_anomaly_callback(alert: dict):
    state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
    machine_id = alert.get("machine_id", "PUMP-001")
    
    # HITL: Persist the incident instead of auto-invoking the agent
    db = SessionLocal()
    try:
        record = AnomalyRecord(
            machine_id=machine_id,
            timestamp=alert.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            type=state,
            score=int(alert.get("reconstruction_score", 0.1) * 100), # Scale to 0-100 for UI
            sensor_data=json.dumps(alert.get("sensor_readings", {}))
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # Broadcast the minimal alert to update the UI Archive
        full_alert_data = {
            "id": record.id,
            "machine_id": machine_id,
            "machine_state": state,
            "anomaly_score": alert.get("reconstruction_score", 0.1),
            "suspect_sensor": alert.get("suspect_sensor"),
            "recent_readings": alert.get("sensor_readings", {})
        }
        
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(async_broadcast_anomaly(full_alert_data))
        except RuntimeError:
            asyncio.run(async_broadcast_anomaly(full_alert_data))
            
    except Exception as e:
        logging.error(f"Failed to persist anomaly: {e}")
    finally:
        db.close()

# Registry of stateful trackers (isolates consecutive counts per machine)
anomaly_trackers: Dict[str, AnomalyService] = {}

def get_tracker(machine_id: str) -> AnomalyService:
    if machine_id not in anomaly_trackers:
        anomaly_trackers[machine_id] = AnomalyService(
            consecutive_threshold=3, 
            on_anomaly=on_anomaly_callback
        )
    return anomaly_trackers[machine_id]

@app.post("/api/telemetry/push")
async def push_telemetry(data: dict, background_tasks: BackgroundTasks):
    """
    Broadcasts live telemetry to WebSockets and asynchronously runs Anomaly Detection.
    """
    machine_id = data.get("machine_id", "PUMP-001")
    tracker = get_tracker(machine_id)
    
    # Run synchronously for broadcast extraction
    result = tracker.process(data)
    
    # Broadcast with health score
    broadcast_payload = {
        "type": "telemetry",
        "data": {
            **data,
            "health_score": result.get("health_score", 100)
        }
    }
    await telemetry_manager.broadcast(json.dumps(broadcast_payload))
    
    return {"status": "broadcast_successful_and_analyzing", "health": result.get("health_score")}

import subprocess
import sys

# Management of multiple independent simulator processes
simulator_processes: Dict[str, subprocess.Popen] = {}

@app.post("/api/simulator/start")
async def start_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    if machine_id not in simulator_processes or simulator_processes[machine_id].poll() is not None:
        # Start new process for the specific machine
        proc = subprocess.Popen([sys.executable, "-m", "simulator.sensor_simulator", "--machine_id", machine_id])
        simulator_processes[machine_id] = proc
        logging.info(f"🚀 Started simulator for {machine_id} (PID: {proc.pid})")
        return {"status": "started", "machine_id": machine_id, "pid": proc.pid}
    return {"status": "already_running", "machine_id": machine_id}

@app.post("/api/simulator/stop")
async def stop_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    if machine_id in simulator_processes and simulator_processes[machine_id].poll() is None:
        proc = simulator_processes[machine_id]
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        del simulator_processes[machine_id]
        logging.info(f"🛑 Stopped simulator for {machine_id}")
        return {"status": "stopped", "machine_id": machine_id}
    return {"status": "not_running", "machine_id": machine_id}

@app.post("/api/simulator/inject")
async def inject_simulator(machine_id: str = "PUMP-001", anomaly_type: str = "machine_fault"):
    state_file = f"simulator_{machine_id}_override.state"
    with open(state_file, "w") as f:
         f.write(anomaly_type)
    logging.info(f"💉 Injecting forced {anomaly_type} for {machine_id}")
    return {"status": "injected", "machine_id": machine_id, "anomaly_type": anomaly_type}

@app.get("/api/simulator/status")
async def get_simulator_status():
    """Returns a list of machine IDs that are currently being simulated."""
    active = [mid for mid, proc in simulator_processes.items() if proc.poll() is None]
    return {"active_simulators": active}

# --- Modular Machine Registry ---
from api.machine_api import router as machine_registry_router
app.include_router(machine_registry_router)

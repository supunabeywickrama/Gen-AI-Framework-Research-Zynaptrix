import logging
import subprocess
import sys
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables early
load_dotenv()

from agents.copilot_graph import build_copilot_graph
from unified_rag.api.endpoints import router as rag_router
from unified_rag.db.models import Machine, AnomalyRecord, ChatMessage, InteractionMemory
from unified_rag.db.database import engine, Base, SessionLocal
from services.anomaly_service import AnomalyService

logging.basicConfig(level=logging.INFO)

import time

try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
except Exception as e:
    logging.warning(f"Vector extension issue: {e}")

# Robust initialization for Neon DB connection drops
max_retries = 3
for attempt in range(max_retries):
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("✅ Database tables verified successfully.")
        break
    except Exception as e:
        logging.error(f"⚠️ Database connection exception during init (attempt {attempt+1}/{max_retries}): {e}")
        if attempt < max_retries - 1:
            time.sleep(2)
        else:
            logging.critical("❌ Could not connect to the database after retries. Starting anyway, some features may fail.")

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


class AnomalyEvent(BaseModel):
    machine_id: str = "PUMP-001"
    machine_state: str = "manual_inquiry"
    anomaly_id: Optional[int] = None
    anomaly_score: Optional[float] = 0.0
    user_query: Optional[str] = None
    suspect_sensor: Optional[str] = "Unknown"
    recent_readings: Optional[Dict[str, Any]] = None


@app.post("/api/copilot/invoke")
def invoke_copilot(event: AnomalyEvent):
    """Synchronous entry point to avoid blocking event loop during heavy RAG."""
    with open("api_debug.log", "a") as f:
        f.write(f"\n--- INQUIRY START: Anomaly #{event.anomaly_id} ---\n")
        # 🧠 PHASE 2: Context Hydration (Fetch PREVIOUS history before adding CURRENT message)
        chat_context = ""
        try:
            db_history = SessionLocal()
            try:
                # Calculate anomaly lookup ID
                look_id = int(event.anomaly_id) if event.anomaly_id else None
                
                past_messages = db_history.query(ChatMessage).filter(
                    ChatMessage.anomaly_id == look_id
                ).order_by(ChatMessage.id).all()
                
                for m in past_messages:
                    chat_context += f"{m.role.upper()}: {m.content}\n"
                
                f.write(f"CONTEXT: Hydrated {len(past_messages)} previous messages for incident {look_id}.\n")
            except Exception as e:
                f.write(f"HISTORY_WARNING: DB offline, providing zero-shot context only: {e}\n")
            finally:
                db_history.close()
        except:
            pass

        # 📝 PHASE 3: Persistent User Message (Store CURRENT after history fetch)
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
                    # Save user message with optional action metadata if it was a button click
                    metadata = None
                    if "[CONVERSATIONAL_WIZARD]" in event.user_query:
                        # Extract intent if it was passed in parentheses (Context: ...)
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
                    db_user.add(msg)
                    db_user.commit()
            except Exception as e:
                f.write(f"DB_WARNING[USER_PHASE]: Connection failed: {e}\n")
            finally:
                db_user.close()
        except Exception as e:
            f.write(f"DB_CRITICAL[INIT]: SessionLocal failed: {e}\n")

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


async def async_broadcast_anomaly(result):
    await telemetry_manager.broadcast(json.dumps({
        "type": "anomaly_alert",
        "data": result
    }))


def on_anomaly_callback(alert: dict):
    """
    Callback when anomaly is detected. Creates AnomalyRecord and runs AI validation.
    """
    state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
    machine_id = alert.get("machine_id", "PUMP-001")
    sensor_readings = alert.get("sensor_readings", {})
    anomaly_score = alert.get("reconstruction_score", 0.1)

    db = SessionLocal()
    try:
        record = AnomalyRecord(
            machine_id=machine_id,
            timestamp=alert.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            type=state,
            score=int(anomaly_score * 100),
            sensor_data=json.dumps(sensor_readings)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        
        # AI Validation Layer: Run validation and persist results
        try:
            from services.sensor_config_loader import sensor_config_loader
            from services.anomaly_service import TemporalAnalyzer, calculate_hybrid_confidence
            from agents.validation_prompts import build_validation_prompt, get_full_system_prompt
            from openai import OpenAI
            from unified_rag.config import settings
            
            # Get physics violations
            physics_summary = sensor_config_loader.get_violation_summary(machine_id, sensor_readings)
            
            # Simplified temporal pattern for real-time callback
            temporal_pattern = {
                "is_spike": False,
                "is_sustained": True,
                "anomaly_count": 1,
                "trend": "stable",
                "max_rate_of_change": 0.0,
                "suspicious_sensors": []
            }
            
            # Calculate hybrid confidence
            hybrid_confidence = calculate_hybrid_confidence(
                anomaly_score, physics_summary, temporal_pattern
            )
            
            # Auto-classify obvious low-confidence cases
            if hybrid_confidence < 0.2:
                record.ai_validation_status = "SENSOR_GLITCH"
                record.fault_category = None
                record.ai_confidence_score = 0.85
                record.ai_engineering_notes = "Low hybrid confidence indicates transient noise. No action required."
                db.commit()
                logging.info(f"⚡ [Validation] Auto-classified anomaly {record.id} as SENSOR_GLITCH")
            else:
                # Call GPT-4o for intelligent validation
                openai_client = OpenAI(api_key=settings.openai_api_key)
                
                prompt = build_validation_prompt(
                    machine_id=machine_id,
                    ml_score=anomaly_score,
                    physics_summary=physics_summary,
                    temporal_pattern=temporal_pattern,
                    recent_readings=sensor_readings,
                    hybrid_confidence=hybrid_confidence
                )
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": get_full_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                validation_json = json.loads(response.choices[0].message.content)
                
                record.ai_validation_status = validation_json.get("ai_validation_status", "TRUE_FAULT")
                record.fault_category = validation_json.get("fault_category")
                record.ai_confidence_score = validation_json.get("confidence_score", hybrid_confidence)
                record.ai_engineering_notes = validation_json.get("ai_engineering_notes", "Validation completed.")
                db.commit()
                
                logging.info(f"✅ [Validation] Anomaly {record.id}: {record.ai_validation_status} | {record.fault_category or 'N/A'} | Confidence: {record.ai_confidence_score:.2f}")
                
        except Exception as validation_error:
            logging.warning(f"⚠️ [Validation] AI validation failed for anomaly {record.id}: {validation_error}")
            # Set fallback values
            record.ai_validation_status = "TRUE_FAULT"
            record.fault_category = "mechanical"
            record.ai_confidence_score = anomaly_score
            record.ai_engineering_notes = "AI validation pending - manual verification recommended."
            db.commit()

        full_alert_data = {
            "id": record.id,
            "machine_id": machine_id,
            "machine_state": state,
            "anomaly_score": anomaly_score,
            "suspect_sensor": alert.get("suspect_sensor"),
            "recent_readings": sensor_readings,
            # Include AI validation in broadcast
            "ai_validation_status": record.ai_validation_status,
            "fault_category": record.fault_category,
            "ai_confidence_score": record.ai_confidence_score,
            "ai_engineering_notes": record.ai_engineering_notes
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
    """Broadcasts live telemetry to WebSockets and asynchronously runs Anomaly Detection."""
    machine_id = data.get("machine_id", "PUMP-001")
    tracker = get_tracker(machine_id)

    try:
        result = tracker.process(data)
        health_score = result.get("health_score", 100)
    except Exception as e:
        logging.error(f"⚠️ Anomaly analysis failed for {machine_id}: {e}")
        health_score = 100  # Fallback

    broadcast_payload = {
        "type": "telemetry",
        "data": {
            **data,
            "health_score": health_score
        }
    }
    await telemetry_manager.broadcast(json.dumps(broadcast_payload))
    return {"status": "broadcast_successful_and_analyzing", "health": health_score}


# ── Simulator management ──────────────────────────────────────────────────────
simulator_processes: Dict[str, subprocess.Popen] = {}


@app.post("/api/simulator/start")
async def start_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    if machine_id not in simulator_processes or simulator_processes[machine_id].poll() is not None:
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


@app.get("/api/machines/{machine_id}/config")
async def get_machine_config(machine_id: str):
    """Return the registered sensor IDs + icon_type for this machine, or defaults."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(backend_dir, "data", "processed", "sensor_configs.json")
    logging.info(f"📂 Config lookup: {config_path}")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            all_configs = json.load(f)
            if machine_id in all_configs:
                machine_cfg = all_configs[machine_id]
                # Build a list of {sensor_id, sensor_name, icon_type, unit}
                sensors_meta = [
                    {
                        "sensor_id":   sid,
                        "sensor_name": sdata.get("sensor_name", sid),
                        "icon_type":   sdata.get("icon_type", "generic"),
                        "unit":        sdata.get("unit", "units"),
                    }
                    for sid, sdata in machine_cfg.items()
                ]
                logging.info(f"✅ Found config for {machine_id}: {[s['sensor_id'] for s in sensors_meta]}")
                return {"sensors": [s["sensor_id"] for s in sensors_meta], "sensors_meta": sensors_meta}

    # Fallback to defaults
    from simulator.anomaly_injector import get_machine_config as get_default
    default_cfg = get_default(machine_id)
    sensors_meta = [
        {"sensor_id": sid, "sensor_name": sid, "icon_type": "generic", "unit": "units"}
        for sid in default_cfg.keys()
    ]
    return {"sensors": list(default_cfg.keys()), "sensors_meta": sensors_meta}


# ── Modular Machine Registry ──────────────────────────────────────────────────
from api.machine_api import router as machine_registry_router
from api.assistant_api import router as assistant_router
from api.evaluation_routes import router as evaluation_router
app.include_router(machine_registry_router)
app.include_router(assistant_router)
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Model Evaluation"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

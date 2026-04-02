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

        initial_state = {
            "event_id": f"EVT-{event.anomaly_id}" if event.anomaly_id else "EVT-LIVE-QUERY",
            "machine_id": event.machine_id,
            "machine_state": event.machine_state,
            "anomaly_score": event.anomaly_score or 0.0,
            "user_query": event.user_query,
            "suspect_sensor": event.suspect_sensor,
            "recent_readings": event.recent_readings or {},
        }

        # Phase 1: Persist user message
        actual_id = None
        db_user = SessionLocal()
        try:
            if event.anomaly_id:
                try:
                    look_id = int(event.anomaly_id)
                    exists = db_user.query(AnomalyRecord).filter(AnomalyRecord.id == look_id).first()
                    if exists:
                        actual_id = look_id
                        f.write(f"VERIFIED: Anomaly #{actual_id} found in DB.\n")
                    else:
                        f.write(f"WARNING: Anomaly #{look_id} NOT in DB.\n")
                except Exception as e:
                    f.write(f"ID_CAST_ERROR: {e}\n")

            if event.user_query:
                msg = ChatMessage(
                    anomaly_id=actual_id,
                    role='user',
                    content=event.user_query,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                db_user.add(msg)
                db_user.commit()
                f.write(f"SUCCESS: User message stored for ID {actual_id}.\n")
        except Exception as e:
            f.write(f"DB_ERROR[USER_PHASE]: {e}\n")
        finally:
            db_user.close()

        # Phase 2: Agent Orchestration
        try:
            result = copilot_workflow.invoke(initial_state)
            f.write("SUCCESS: Graph execution completed.\n")
        except Exception as e:
            f.write(f"GRAPH_ERROR: {e}\n")
            return {"status": "error", "message": f"Orchestration failure: {str(e)}"}

        # Phase 3: Persist agent response
        final_answer = result.get("final_execution_plan", "")
        if final_answer:
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
                f.write(f"DB_ERROR[AGENT_PHASE]: {e}\n")
            finally:
                db_agent.close()

        return {"status": "success", "graph_result": result, "stored_id": actual_id}


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
    state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
    machine_id = alert.get("machine_id", "PUMP-001")

    db = SessionLocal()
    try:
        record = AnomalyRecord(
            machine_id=machine_id,
            timestamp=alert.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            type=state,
            score=int(alert.get("reconstruction_score", 0.1) * 100),
            sensor_data=json.dumps(alert.get("sensor_readings", {}))
        )
        db.add(record)
        db.commit()
        db.refresh(record)

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
app.include_router(machine_registry_router)

import logging
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

from agents.copilot_graph import build_copilot_graph
from unified_rag.api.endpoints import router as rag_router
from unified_rag.db.database import engine, Base
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

class AnomalyEvent(BaseModel):
    machine_id: str = "PUMP-001"
    machine_state: str
    anomaly_score: float
    suspect_sensor: Optional[str] = "Unknown"
    recent_readings: Optional[Dict[str, Any]] = None

@app.post("/api/copilot/invoke")
async def invoke_copilot(event: AnomalyEvent):
    initial_state = {
        "event_id": "EVT-LIVE-ALERT",
        "machine_id": event.machine_id,
        "machine_state": event.machine_state,
        "anomaly_score": event.anomaly_score,
        "suspect_sensor": event.suspect_sensor,
        "recent_readings": event.recent_readings or {},
    }
    result = copilot_workflow.invoke(initial_state)
    return {"status": "success", "graph_result": result}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await telemetry_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)
        
# --- Real-Time Agentic Orchestration Layer ---

async def async_broadcast_anomaly(result):
    await telemetry_manager.broadcast(json.dumps({
        "type": "anomaly_alert", 
        "data": result
    }))

def trigger_orchestrator(alert_data):
    initial_state = {
        "event_id": "EVT-CRITICAL-ALERT",
        "machine_id": alert_data.get("machine_id", "PUMP-001"),
        "machine_state": alert_data["machine_state"],
        "anomaly_score": alert_data["anomaly_score"],
        "suspect_sensor": alert_data["suspect_sensor"],
        "recent_readings": alert_data["recent_readings"],
    }
    result = copilot_workflow.invoke(initial_state)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(async_broadcast_anomaly(result))
    except RuntimeError:
        asyncio.run(async_broadcast_anomaly(result))

def on_anomaly_callback(alert: dict):
    state = "machine_fault" if alert.get("severity") == "HIGH" else "machine_warning"
    full_alert_data = {
        "machine_id": alert.get("machine_id", "PUMP-001"),
        "machine_state": state,
        "anomaly_score": alert.get("reconstruction_score", 0.0),
        "suspect_sensor": alert.get("suspect_sensor"),
        "recent_readings": alert.get("sensor_readings", {})
    }
    trigger_orchestrator(full_alert_data)

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
    await telemetry_manager.broadcast(json.dumps({"type": "telemetry", "data": data}))
    
    machine_id = data.get("machine_id", "PUMP-001")
    tracker = get_tracker(machine_id)
    
    def background_anomaly_check(reading, t):
        t.process(reading)
        
    background_tasks.add_task(background_anomaly_check, data, tracker)
    return {"status": "broadcast_successful_and_analyzing"}

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

@app.get("/api/simulator/status")
async def get_simulator_status():
    """Returns a list of machine IDs that are currently being simulated."""
    active = [mid for mid, proc in simulator_processes.items() if proc.poll() is None]
    return {"active_simulators": active}

# --- Modular Machine Registry ---
from api.machine_api import router as machine_registry_router
app.include_router(machine_registry_router)

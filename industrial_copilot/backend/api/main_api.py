import logging
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json
import os

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
    machine_state: str
    anomaly_score: float
    suspect_sensor: Optional[str] = "Unknown"
    recent_readings: Optional[Dict[str, Any]] = None

@app.post("/api/copilot/invoke")
async def invoke_copilot(event: AnomalyEvent):
    initial_state = {
        "event_id": "EVT-LIVE-ALERT",
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
        "machine_state": state,
        "anomaly_score": alert.get("reconstruction_score", 0.0),
        "suspect_sensor": alert.get("suspect_sensor"),
        "recent_readings": alert.get("sensor_readings", {})
    }
    trigger_orchestrator(full_alert_data)

anomaly_tracker = AnomalyService(consecutive_threshold=3, on_anomaly=on_anomaly_callback)

@app.post("/api/telemetry/push")
async def push_telemetry(data: dict, background_tasks: BackgroundTasks):
    """
    Broadcasts live telemetry to WebSockets and asynchronously runs Anomaly Detection.
    If an anomaly is triggered 3 consecutive ticks, the Orchestrator will seamlessly run
    in an external thread and multiplex back an Agentic Procedure payload over WebSockets!
    """
    await telemetry_manager.broadcast(json.dumps({"type": "telemetry", "data": data}))
    
    def background_anomaly_check(reading):
        anomaly_tracker.process(reading)
        
    background_tasks.add_task(background_anomaly_check, data)
    return {"status": "broadcast_successful_and_analyzing"}

import subprocess
import sys

simulator_process = None

@app.post("/api/simulator/start")
async def start_simulator():
    global simulator_process
    if simulator_process is None or simulator_process.poll() is not None:
        simulator_process = subprocess.Popen([sys.executable, "-m", "simulator.sensor_simulator"])
        return {"status": "started", "pid": simulator_process.pid}
    return {"status": "already_running"}

@app.post("/api/simulator/stop")
async def stop_simulator():
    global simulator_process
    if simulator_process and simulator_process.poll() is None:
        simulator_process.terminate()
        try:
            simulator_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            simulator_process.kill()
        simulator_process = None
        return {"status": "stopped"}
    return {"status": "not_running"}

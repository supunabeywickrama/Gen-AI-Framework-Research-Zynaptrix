import logging
from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import json

from agents.copilot_graph import build_copilot_graph
from api.rag_routes import router as rag_router
from rag_db.database import engine, Base
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)

try:
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
except Exception as e:
    logging.warning(f"Vector extension issue: {e}")

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Generative AI Multi-Agent Industrial Copilot")

app.include_router(rag_router, tags=["Knowledge Base"])

# Ensure Next.js Frontend can securely communicate with Backend
import os
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
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
    """
    Triggers the LangGraph multi-agent orchestration dynamically.
    Sequentially executes: Sensor -> Diagnostic -> Retrieval -> Strategy -> Critic.
    Returns the accumulated CopilotState containing all reports and final execution plan.
    """
    initial_state = {
        "event_id": "EVT-LIVE-ALERT",
        "machine_state": event.machine_state,
        "anomaly_score": event.anomaly_score,
        "suspect_sensor": event.suspect_sensor,
        "recent_readings": event.recent_readings or {},
    }
    
    # Run the Graph logic
    result = copilot_workflow.invoke(initial_state)
    return {"status": "success", "graph_result": result}

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """
    Provides real-time IoT sensor telemetry streams to the Next.js Dashboard.
    """
    await telemetry_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, listen for client messages if any
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        telemetry_manager.disconnect(websocket)
        
@app.post("/api/telemetry/push")
async def push_telemetry(data: dict):
    """
    Called by the InfluxDB simulator/stream_listener to securely push 
    live data to connected Next.js dashboard WebSockets without polling.
    """
    await telemetry_manager.broadcast(json.dumps(data))
    return {"status": "broadcast_successful"}

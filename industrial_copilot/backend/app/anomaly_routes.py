"""
anomaly_routes.py — Endpoints for anomaly detection and history retrieval.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pydantic import BaseModel
from datetime import datetime, timezone
import logging

from services.monitoring_service import MonitoringService
from database.neon_vector_store import NeonVectorStore

log = logging.getLogger(__name__)
router = APIRouter()

# Instantiate global service
monitor_service = MonitoringService()

LATEST_READING_STATE = {
    "reading": None,
    "is_anomaly": False,
    "escalated": False,
    "score": 0.0,
    "threshold": 0.7187
}

class SensorReading(BaseModel):
    temperature: float
    motor_current: float
    vibration: float
    speed: float
    pressure: float

@router.post("/detect")
async def detect_anomaly(reading: SensorReading):
    """
    Process a single sensor reading through the anomaly detection pipeline.
    If it triggers a sustained fault, the OrchestratorAgent automatically intervenes.
    """
    try:
        reading_dict = reading.model_dump()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Track in anomaly service (this calls _on_anomaly_confirmed if threshold met)
        result = monitor_service.process_reading(reading_dict, timestamp)
        
        # update global cache for passive UI listeners
        LATEST_READING_STATE.update({
            "reading": reading_dict,
            "is_anomaly": result.get("is_anomaly", False),
            "escalated": result.get("escalated", False),
            "score": result.get("score"),
            "threshold": result.get("threshold")
        })
        
        return {
            "status": "success", 
            "is_anomaly": LATEST_READING_STATE["is_anomaly"],
            "escalated": LATEST_READING_STATE["escalated"],
            "score": LATEST_READING_STATE["score"],
            "threshold": LATEST_READING_STATE["threshold"]
        }
        
    except Exception as e:
        log.error(f"Error during anomaly detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(limit: int = Query(20, ge=1, le=100)):
    """
    Retrieve the most recent anomaly logs resolved by the Orchestrator.
    """
    try:
        store = NeonVectorStore()
        events = store.get_recent_events(limit=limit)
        
        # JSON serialize datetime objects safely
        for event in events:
            if 'event_time' in event and event['event_time']:
                event['event_time'] = event['event_time'].isoformat()
            if 'resolved_at' in event and event['resolved_at']:
                event['resolved_at'] = event['resolved_at'].isoformat()
                
        return {"status": "success", "events": events}
    except Exception as e:
        log.error(f"Database error fetching anomaly history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history from database.")

@router.get("/latest")
async def get_latest_reading():
    """
    Returns the most recently processed stream reading from InfluxDB.
    Used by the Streamlit UI for completely decoupled, passive live-monitoring.
    """
    return {"status": "success", "data": LATEST_READING_STATE}

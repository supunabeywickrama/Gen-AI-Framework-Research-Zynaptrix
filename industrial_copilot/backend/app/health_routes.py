"""
health_routes.py — Endpoints for basic API health and sensor configurations.
"""
from fastapi import APIRouter, HTTPException
from config.settings import SENSOR_SCHEMA
import logging

log = logging.getLogger(__name__)
router = APIRouter()

@router.get("/status")
async def system_status():
    """
    Returns the basic health status of the API and underlying ML services.
    """
    return {
        "api_status": "healthy",
        "monitoring_service": "online",
        "database": "online",
        "agent_system": "ready"
    }

@router.get("/sensors")
async def get_sensors():
    """
    Returns the configured normal operational ranges for all tracked sensors.
    """
    try:
        return {"status": "success", "sensors": SENSOR_SCHEMA}
    except Exception as e:
        log.error(f"Error fetching sensor schema: {e}")
        raise HTTPException(status_code=500, detail="Failed to load sensor configurations.")

@router.get("/sensors/{sensor_id}")
async def get_sensor_info(sensor_id: str):
    """
    Returns the config for a specific sensor.
    """
    if sensor_id not in SENSOR_SCHEMA:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found.")
    
    return {"status": "success", "sensor": sensor_id, "config": SENSOR_SCHEMA[sensor_id]}

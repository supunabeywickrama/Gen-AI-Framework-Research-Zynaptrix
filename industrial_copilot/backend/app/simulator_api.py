import os
import sys
import json
import logging
import subprocess
from typing import Dict, List
from fastapi import APIRouter

from simulator.anomaly_injector import get_machine_config as get_default

router = APIRouter(prefix="/api/simulator", tags=["Simulator"])
logger = logging.getLogger(__name__)

simulator_processes: Dict[str, subprocess.Popen] = {}

@router.post("/start")
async def start_simulator(machine_id: str = "PUMP-001"):
    global simulator_processes
    if machine_id not in simulator_processes or simulator_processes[machine_id].poll() is not None:
        proc = subprocess.Popen([sys.executable, "-m", "simulator.sensor_simulator", "--machine_id", machine_id])
        simulator_processes[machine_id] = proc
        logger.info(f"🚀 Started simulator for {machine_id} (PID: {proc.pid})")
        return {"status": "started", "machine_id": machine_id, "pid": proc.pid}
    return {"status": "already_running", "machine_id": machine_id}

@router.post("/stop")
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
        logger.info(f"🛑 Stopped simulator for {machine_id}")
        return {"status": "stopped", "machine_id": machine_id}
    return {"status": "not_running", "machine_id": machine_id}

@router.post("/inject")
async def inject_simulator(machine_id: str = "PUMP-001", anomaly_type: str = "machine_fault"):
    state_file = f"simulator_{machine_id}_override.state"
    with open(state_file, "w") as f:
         f.write(anomaly_type)
    logger.info(f"💉 Injecting forced {anomaly_type} for {machine_id}")
    return {"status": "injected", "machine_id": machine_id, "anomaly_type": anomaly_type}

@router.get("/status")
async def get_simulator_status():
    """Returns a list of machine IDs that are currently being simulated."""
    active = [mid for mid, proc in simulator_processes.items() if proc.poll() is None]
    return {"active_simulators": active}

@router.get("/machine/{machine_id}/config")
async def get_machine_config(machine_id: str):
    """Return the registered sensor IDs + icon_type for this machine, or defaults."""
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(backend_dir, "data", "processed", "sensor_configs.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                all_configs = json.load(f)
                if machine_id in all_configs:
                    machine_cfg = all_configs[machine_id]
                    sensors_meta = [
                        {
                            "sensor_id":   sid,
                            "sensor_name": sdata.get("sensor_name", sid),
                            "icon_type":   sdata.get("icon_type", "generic"),
                            "unit":        sdata.get("unit", "units"),
                        }
                        for sid, sdata in machine_cfg.items()
                    ]
                    return {"sensors": [s["sensor_id"] for s in sensors_meta], "sensors_meta": sensors_meta}
        except Exception as e:
            logger.error(f"Error reading sensor configs: {e}")

    # Fallback to defaults
    default_cfg = get_default(machine_id)
    sensors_meta = [
        {"sensor_id": sid, "sensor_name": sid, "icon_type": "generic", "unit": "units"}
        for sid in default_cfg.keys()
    ]
    return {"sensors": list(default_cfg.keys()), "sensors_meta": sensors_meta}

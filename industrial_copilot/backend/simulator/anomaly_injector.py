"""
anomaly_injector.py — State-controlled anomaly injection logic.

Provides functions that return sensor readings for each machine state.
Used by both the batch dataset generator and the real-time simulator.
"""

import numpy as np


import os
import json

def get_machine_config(machine_id: str) -> dict:
    """Returns base sensor ranges and noise profiles for specific machines either from dynamic JSON or defaults."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sensor_configs.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            all_configs = json.load(f)
            if machine_id in all_configs:
                # all_configs[machine_id] = { "sensor_id": {"sensor_id": "...", "sensor_name": "...", "mu": 50.0, "sigma": 5.0} }
                dyn_cfg = {}
                for sid, data in all_configs[machine_id].items():
                    dyn_cfg[sid] = (data.get("mu", 50.0), data.get("sigma", 5.0))
                if dyn_cfg:
                    return dyn_cfg
                    
    # Fallback Defaults
    if "LATHE" in machine_id.upper():
        return {
            "temperature":   (45, 2),    # Spindle Temperature
            "motor_current": (12.5, 1.5),# Spindle Load (Amps)
            "vibration":     (0.15, 0.05),# Tool Vibration
            "speed":         (3200, 20), # Spindle RPM
            "pressure":      (8.5, 0.5),  # Coolant Pressure
        }
    elif "TURBINE" in machine_id.upper():
        return {
            "temperature":   (850, 15),  # Core Temperature
            "motor_current": (450, 5.0), # Generator Load (Amps) - Scaled
            "vibration":     (1.2, 0.2), # Rotor Vibration
            "speed":         (15000, 50),# Rotor RPM
            "pressure":      (32.0, 1.0), # Fuel Rail Pressure
        }
    else: # Default (PUMP-001)
        return {
            "temperature":   (180, 2),
            "motor_current": (4.5, 0.5),
            "vibration":     (0.8, 0.1),
            "speed":         (160, 5),
            "pressure":      (4.5, 0.2),
        }

def normal_reading(machine_id: str = "PUMP-001") -> dict:
    """Stable readings with small Gaussian noise — healthy machine."""
    cfg = get_machine_config(machine_id)
    return {k: np.random.normal(mu, sigma) for k, (mu, sigma) in cfg.items()}

def machine_fault_reading(machine_id: str = "PUMP-001") -> dict:
    """
    Simulates a mechanical fault by shifting means by an unstable variance.
    """
    cfg = get_machine_config(machine_id)
    fault_reading = {}
    for k, (mu, sigma) in cfg.items():
        if "temp" in k.lower() or "vibration" in k.lower() or "current" in k.lower():
            fault_reading[k] = np.random.normal(mu * 1.5, sigma * 3)
        else:
            fault_reading[k] = np.random.normal(mu * 0.8, sigma * 2)
    return fault_reading

def sensor_freeze_reading(frozen_values: dict | None = None) -> dict:
    """Simulates a stuck/frozen sensor."""
    if frozen_values is None:
        frozen_values = normal_reading("PUMP-001")
    return {k: v for k, v in frozen_values.items()}

def sensor_drift_reading(drift_step: float = 0.0, machine_id: str = "PUMP-001") -> dict:
    """Simulates the first sensor gradually drifting upward."""
    base = normal_reading(machine_id)
    # Pick the first key as the target for drift
    drift_key = list(base.keys())[0] if base else "temperature"
    if drift_key in base:
        base[drift_key] += drift_step
    return base

def idle_reading(machine_id: str = "PUMP-001") -> dict:
    """Machine is powered off / idle."""
    cfg = get_machine_config(machine_id)
    idle = {}
    for k, (mu, sigma) in cfg.items():
        if "temp" in k.lower():
            idle[k] = np.random.normal(25, 1) # ambient temp
        else:
            idle[k] = 0.0
    return idle

STATE_GENERATORS = {
    "normal":        normal_reading,
    "machine_fault": machine_fault_reading,
    "sensor_freeze": sensor_freeze_reading,
    "idle":          idle_reading,
}

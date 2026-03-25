"""
anomaly_injector.py — State-controlled anomaly injection logic.

Provides functions that return sensor readings for each machine state.
Used by both the batch dataset generator and the real-time simulator.
"""

import numpy as np


def get_machine_config(machine_id: str) -> dict:
    """Returns base sensor ranges and noise profiles for specific machines."""
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
    Simulates a mechanical fault (e.g. motor bearing wear).
    """
    cfg = get_machine_config(machine_id)
    # Increase vibration and current, drop speed
    return {
        "temperature":   np.random.normal(cfg["temperature"][0] - 5, cfg["temperature"][1]),
        "motor_current": np.random.normal(cfg["motor_current"][0] * 1.6, cfg["motor_current"][1] * 2),
        "vibration":     np.random.normal(cfg["vibration"][0] * 3.5, cfg["vibration"][1] * 5),
        "speed":         np.random.normal(cfg["speed"][0] * 0.8, cfg["speed"][1] * 2),
        "pressure":      np.random.normal(cfg["pressure"][0] * 0.9, cfg["pressure"][1]),
    }

def sensor_freeze_reading(frozen_values: dict | None = None) -> dict:
    """Simulates a stuck/frozen sensor."""
    if frozen_values is None:
        frozen_values = normal_reading("PUMP-001")
    return {k: v for k, v in frozen_values.items()}

def sensor_drift_reading(drift_step: float = 0.0, machine_id: str = "PUMP-001") -> dict:
    """Simulates a temperature sensor that gradually drifts upward."""
    base = normal_reading(machine_id)
    base["temperature"] += drift_step
    return base

def idle_reading() -> dict:
    """Machine is powered off / idle."""
    return {
        "temperature":   np.random.normal(25, 1),
        "motor_current": 0.0,
        "vibration":     0.0,
        "speed":         0.0,
        "pressure":      0.0,
    }

STATE_GENERATORS = {
    "normal":        normal_reading,
    "machine_fault": machine_fault_reading,
    "sensor_freeze": sensor_freeze_reading,
    "idle":          idle_reading,
}

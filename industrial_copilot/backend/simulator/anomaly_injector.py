"""
anomaly_injector.py — State-controlled anomaly injection logic.

Provides functions that return sensor readings for each machine state.
Used by both the batch dataset generator and the real-time simulator.
"""

import numpy as np


def normal_reading() -> dict:
    """Stable readings with small Gaussian noise — healthy machine."""
    return {
        "temperature":   np.random.normal(180, 2),
        "motor_current": np.random.normal(4.5, 0.5),
        "vibration":     np.random.normal(0.8, 0.1),
        "speed":         np.random.normal(160, 5),
        "pressure":      np.random.normal(4.5, 0.2),
    }


def machine_fault_reading() -> dict:
    """
    Simulates a mechanical fault (e.g. motor bearing wear).
    Characteristics:
      - vibration spikes
      - motor current increases (overload)
      - speed drops
      - temperature slightly lower (machine slowing down)
    """
    return {
        "temperature":   np.random.normal(175, 3),
        "motor_current": np.random.normal(7.0, 1.0),
        "vibration":     np.random.normal(2.5, 0.5),
        "speed":         np.random.normal(135, 10),
        "pressure":      np.random.normal(4.2, 0.3),
    }


def sensor_freeze_reading(frozen_values: dict | None = None) -> dict:
    """
    Simulates a stuck/frozen sensor.
    All sensors are frozen at the same constant value.
    Other sensors may drift slightly to simulate the difference.
    """
    if frozen_values is None:
        frozen_values = {
            "temperature":   180.0,
            "motor_current": 4.5,
            "vibration":     0.83,
            "speed":         160.0,
            "pressure":      4.5,
        }
    return {k: v for k, v in frozen_values.items()}


def sensor_drift_reading(drift_step: float = 0.0) -> dict:
    """
    Simulates a temperature sensor that gradually drifts upward.
    Other sensors remain normal.
    drift_step: cumulative drift offset added to temperature.
    """
    return {
        "temperature":   180.0 + drift_step + np.random.normal(0, 0.3),
        "motor_current": np.random.normal(4.5, 0.5),
        "vibration":     np.random.normal(0.8, 0.1),
        "speed":         np.random.normal(160, 5),
        "pressure":      np.random.normal(4.5, 0.2),
    }


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

"""
sensor_config_loader.py — Loads and validates sensor physics limits from configuration files.

Purpose: Validates sensor readings against manufacturer datasheets and physics limits
         to distinguish real faults from sensor glitches.

Features:
    - Singleton pattern for efficient memory usage
    - Loads sensor specifications from sensor_configs.json
    - Validates readings against min_normal, max_normal, fault_high, fault_low
    - Returns violation severity and description
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from threading import Lock

logger = logging.getLogger(__name__)


class SensorConfigLoader:
    """
    Singleton class for loading and managing sensor configurations.
    Provides physics-based validation of sensor readings.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._configs: Dict[str, Dict[str, Any]] = {}
        self._load_configs()
        self._initialized = True
    
    def _load_configs(self) -> None:
        """Load sensor configurations from JSON file."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "data", "processed", "sensor_configs.json")
        
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self._configs = json.load(f)
                logger.info(f"✅ Loaded sensor configs for {len(self._configs)} machines")
            else:
                logger.warning(f"⚠️ Sensor config file not found: {config_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load sensor configs: {e}")
    
    def get_machine_config(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Get sensor configuration for a specific machine."""
        return self._configs.get(machine_id)
    
    def check_physics_violation(self, machine_id: str, sensor_id: str, value: float) -> Dict[str, Any]:
        """
        Check if a sensor reading violates physics limits.
        
        Args:
            machine_id: The machine identifier
            sensor_id: The sensor identifier  
            value: The sensor reading value
            
        Returns:
            dict with keys:
                - has_violation: bool
                - severity: 'critical' | 'warning' | 'none'
                - description: str
                - threshold_exceeded: float or None
        """
        machine_config = self._configs.get(machine_id, {})
        sensor_config = machine_config.get(sensor_id, {})
        
        if not sensor_config:
            return {
                "has_violation": False,
                "severity": "none",
                "description": f"No config for {sensor_id}",
                "threshold_exceeded": None
            }
        
        fault_high = sensor_config.get("fault_high")
        fault_low = sensor_config.get("fault_low")
        max_normal = sensor_config.get("max_normal")
        min_normal = sensor_config.get("min_normal")
        unit = sensor_config.get("unit", "units")
        sensor_name = sensor_config.get("sensor_name", sensor_id)
        
        # Check CRITICAL violations (fault thresholds)
        if fault_high is not None and value > fault_high:
            return {
                "has_violation": True,
                "severity": "critical",
                "description": f"{sensor_name}: {value:.2f}{unit} > fault_high ({fault_high}{unit}) - CRITICAL",
                "threshold_exceeded": value - fault_high
            }
        
        if fault_low is not None and value < fault_low:
            return {
                "has_violation": True,
                "severity": "critical", 
                "description": f"{sensor_name}: {value:.2f}{unit} < fault_low ({fault_low}{unit}) - CRITICAL",
                "threshold_exceeded": fault_low - value
            }
        
        # Check WARNING violations (normal range exceeded)
        if max_normal is not None and value > max_normal:
            return {
                "has_violation": True,
                "severity": "warning",
                "description": f"{sensor_name}: {value:.2f}{unit} > max_normal ({max_normal}{unit}) - WARNING",
                "threshold_exceeded": value - max_normal
            }
        
        if min_normal is not None and value < min_normal:
            return {
                "has_violation": True,
                "severity": "warning",
                "description": f"{sensor_name}: {value:.2f}{unit} < min_normal ({min_normal}{unit}) - WARNING", 
                "threshold_exceeded": min_normal - value
            }
        
        return {
            "has_violation": False,
            "severity": "none",
            "description": f"{sensor_name}: {value:.2f}{unit} within normal range",
            "threshold_exceeded": None
        }
    
    def get_violation_summary(self, machine_id: str, readings: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze all sensor readings for a machine and return violation summary.
        
        Args:
            machine_id: The machine identifier
            readings: Dict of sensor_id -> value
            
        Returns:
            dict with keys:
                - has_violations: bool
                - fault_violations: list of critical violations
                - normal_violations: list of warning violations
                - summary_text: str for AI prompt
        """
        fault_violations: List[Dict] = []
        normal_violations: List[Dict] = []
        
        for sensor_id, value in readings.items():
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
                
            result = self.check_physics_violation(machine_id, sensor_id, value_float)
            
            if result["severity"] == "critical":
                fault_violations.append({
                    "sensor_id": sensor_id,
                    "value": value_float,
                    **result
                })
            elif result["severity"] == "warning":
                normal_violations.append({
                    "sensor_id": sensor_id,
                    "value": value_float,
                    **result
                })
        
        # Build summary text for AI prompt
        summary_lines = []
        if fault_violations:
            summary_lines.append("CRITICAL VIOLATIONS:")
            for v in fault_violations:
                summary_lines.append(f"  - {v['description']}")
        if normal_violations:
            summary_lines.append("WARNINGS:")
            for v in normal_violations:
                summary_lines.append(f"  - {v['description']}")
        if not fault_violations and not normal_violations:
            summary_lines.append("All sensors within normal operating range.")
        
        return {
            "has_violations": len(fault_violations) > 0 or len(normal_violations) > 0,
            "fault_violations": fault_violations,
            "normal_violations": normal_violations,
            "summary_text": "\n".join(summary_lines)
        }


# Singleton instance for import convenience
sensor_config_loader = SensorConfigLoader()

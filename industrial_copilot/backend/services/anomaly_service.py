"""
anomaly_service.py — Real-time anomaly detection pipeline service.

Pipeline:
    new sensor reading (dict)
        ↓  normalize
        ↓  AnomalyDetector.detect()
        ↓  if anomaly → alert_service.trigger()
        ↓  log result

Used by:
  - The real-time simulator (sensor_simulator.py)
  - The FastAPI anomaly endpoint (api/anomaly_routes.py)
"""

import logging
from datetime import datetime, timezone
from typing import Callable

from models.detect_anomaly import AnomalyDetector
from services.alert_service import format_alert, log_alert

log = logging.getLogger(__name__)


class AnomalyService:
    """
    Stateful service wrapping AnomalyDetector.
    Maintains a rolling count of recent anomalies and exposes
    a callback hook so callers can react to events.
    """

    def __init__(self,
                 on_anomaly: Callable[[dict], None] | None = None,
                 consecutive_threshold: int = 3):
        """
        Args:
            on_anomaly:              Optional callback called with alert dict on anomaly.
            consecutive_threshold:   Number of consecutive anomaly readings before
                                     escalating (future use by agent layer).
        """
        self._detector   = AnomalyDetector()
        self._on_anomaly = on_anomaly
        self._consec_threshold = consecutive_threshold
        self._consecutive_count = 0
        self._total_processed   = 0
        self._total_anomalies   = 0

    def process(self, reading: dict) -> dict:
        """
        Process one sensor reading through the anomaly detection pipeline.

        Args:
            reading: dict with sensor keys matching SENSOR_COLUMNS.

        Returns:
            result dict from AnomalyDetector.detect() augmented with:
              - timestamp
              - consecutive_anomalies
              - escalated (bool)
        """
        result = self._detector.detect(reading)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        health_score = result.get("health_score", 100)

        self._total_processed += 1

        if result["is_anomaly"]:
            self._consecutive_count += 1
            self._total_anomalies   += 1
            result["consecutive_anomalies"] = self._consecutive_count
            result["escalated"] = self._consecutive_count >= self._consec_threshold

            alert = format_alert(result)
            alert["health_score"] = health_score
            log_alert(alert)

            if self._on_anomaly:
                self._on_anomaly(alert)
        else:
            self._consecutive_count = 0
            result["consecutive_anomalies"] = 0
            result["escalated"] = False

        log.debug(
            f"[{'ANOMALY' if result['is_anomaly'] else 'OK':7s}] "
            f"score={result['score']:.5f}  "
            f"threshold={result['threshold']:.5f}"
        )
        return result

    @property
    def stats(self) -> dict:
        return {
            "total_processed":   self._total_processed,
            "total_anomalies":   self._total_anomalies,
            "anomaly_rate_pct":  round(
                self._total_anomalies / max(self._total_processed, 1) * 100, 2
            ),
        }

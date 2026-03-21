"""
test_model.py — Unit tests for the anomaly detection model pipeline.

Tests:
  1. AnomalyDetector loads without errors
  2. Single normal reading produces score < threshold (low error)
  3. Single fault reading produces score > threshold (high error)
  4. Batch detect returns correct columns and flags known anomalies
  5. AnomalyService stats accumulate correctly

Run:
    python -m pytest tests/test_model.py -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def detector():
    """Load detector once for all tests."""
    from models.detect_anomaly import AnomalyDetector
    d = AnomalyDetector()
    d._load()   # trigger eager load
    return d


@pytest.fixture
def normal_reading():
    return {
        "temperature":   180.2,
        "motor_current": 4.3,
        "vibration":     0.81,
        "speed":         161.0,
        "pressure":      4.4,
    }


@pytest.fixture
def fault_reading():
    """Severe machine fault — vibration and current are way outside normal."""
    return {
        "temperature":   172.0,
        "motor_current": 8.5,
        "vibration":     3.8,
        "speed":         125.0,
        "pressure":      3.9,
    }


@pytest.fixture
def idle_reading():
    return {
        "temperature":   25.0,
        "motor_current": 0.0,
        "vibration":     0.0,
        "speed":         0.0,
        "pressure":      0.0,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_detector_loads(self, detector):
        assert detector.threshold > 0, "Threshold must be positive"

    def test_detect_returns_required_keys(self, detector, normal_reading):
        result = detector.detect(normal_reading)
        for key in ("is_anomaly", "score", "threshold", "sensors"):
            assert key in result, f"Missing key: {key}"

    def test_normal_reading_not_flagged(self, detector, normal_reading):
        result = detector.detect(normal_reading)
        assert not result["is_anomaly"], (
            f"Normal reading flagged as anomaly: score={result['score']:.5f} "
            f"threshold={result['threshold']:.5f}"
        )

    def test_fault_reading_flagged(self, detector, fault_reading):
        result = detector.detect(fault_reading)
        assert result["is_anomaly"], (
            f"Fault reading NOT flagged: score={result['score']:.5f} "
            f"threshold={result['threshold']:.5f}"
        )

    def test_idle_reading_flagged(self, detector, idle_reading):
        result = detector.detect(idle_reading)
        assert result["is_anomaly"], "Idle machine should be flagged as anomaly"

    def test_score_is_float(self, detector, normal_reading):
        result = detector.detect(normal_reading)
        assert isinstance(result["score"], float)

    def test_batch_detect_columns(self, detector):
        from config.settings import SENSOR_COLUMNS, MOCK_DATA_PATH
        df = pd.read_csv(MOCK_DATA_PATH).sample(200, random_state=0)
        result_df = detector.detect_batch(df)
        assert "recon_error"  in result_df.columns
        assert "is_anomaly"   in result_df.columns
        assert len(result_df) == 200

    def test_batch_fault_rows_mostly_flagged(self, detector):
        from config.settings import MOCK_DATA_PATH
        df = pd.read_csv(MOCK_DATA_PATH)
        faults = df[df["state"] == "machine_fault"]
        result_df = detector.detect_batch(faults)
        flagged_pct = result_df["is_anomaly"].mean() * 100
        # At least 60% of fault rows should be caught
        assert flagged_pct >= 60, (
            f"Too few fault rows flagged: {flagged_pct:.1f}%"
        )

    def test_batch_normal_rows_low_false_positive(self, detector):
        from config.settings import MOCK_DATA_PATH
        df = pd.read_csv(MOCK_DATA_PATH)
        normals = df[df["state"] == "normal"].sample(500, random_state=42)
        result_df = detector.detect_batch(normals)
        fp_pct = result_df["is_anomaly"].mean() * 100
        # False positive rate should stay below 10%
        assert fp_pct < 10, f"Too many false positives on normal data: {fp_pct:.1f}%"


class TestAnomalyService:

    def test_service_stats_accumulate(self, detector, normal_reading, fault_reading):
        from services.anomaly_service import AnomalyService
        service = AnomalyService()
        service.process(normal_reading)
        service.process(normal_reading)
        service.process(fault_reading)
        stats = service.stats
        assert stats["total_processed"] == 3
        assert stats["total_anomalies"] >= 1

    def test_callback_called_on_anomaly(self, fault_reading):
        from services.anomaly_service import AnomalyService
        events = []
        service = AnomalyService(on_anomaly=lambda a: events.append(a))
        service.process(fault_reading)
        assert len(events) >= 1, "Callback not triggered on fault reading"

    def test_consecutive_count_resets_on_normal(self, normal_reading, fault_reading):
        from services.anomaly_service import AnomalyService
        service = AnomalyService()
        service.process(fault_reading)
        assert service._consecutive_count >= 1
        service.process(normal_reading)
        assert service._consecutive_count == 0

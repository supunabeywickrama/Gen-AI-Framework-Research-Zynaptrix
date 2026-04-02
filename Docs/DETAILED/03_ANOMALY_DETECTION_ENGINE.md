# Part 3: Anomaly Detection Engine

## 3.1 Overview

The anomaly detection layer is a three-tier stack:
1. **`AnomalyDetector`** (`models/detect_anomaly.py`) — Low-level ML inference: loads the model, normalizes input, runs reconstruction, computes MSE.
2. **`AnomalyService`** (`services/anomaly_service.py`) — Stateful wrapper: tracks consecutive anomaly counts, escalation state, and fires callbacks.
3. **`MonitoringService`** (`services/monitoring_service.py`) — Top-level orchestration: connects `AnomalyService` to the `OrchestratorAgent`.

---

## 3.2 `AnomalyDetector` (`models/detect_anomaly.py`)

### 3.2.1 Design Philosophy

The detector is **machine-aware**: it maintains an internal registry of models and scalers indexed by `machine_id`. When a reading arrives for `LATHE-002`, it loads and caches the `LATHE-002` Autoencoder and its corresponding `scaler_LATHE-002.pkl`. This prevents cross-contamination — a PUMP normal reading would appear anomalous to a TURBINE model.

### 3.2.2 Model Registry Pattern

```python
self._model_cache  = {}   # machine_id → Keras model
self._scaler_cache = {}   # machine_id → StandardScaler
```

On first call, the model is loaded from disk and cached in memory. Subsequent calls for the same machine skip disk I/O entirely. This is critical for a 10Hz inference loop where model loading latency would be catastrophic.

### 3.2.3 Model File Path Convention

Models are stored at:
```
data/processed/autoencoder_{machine_id}.keras
data/processed/scaler_{machine_id}.pkl
```

The `AnomalyDetector` dynamically constructs these paths using the `machine_id` string. If a machine-specific model doesn't exist, it falls back to the default `PUMP-001` model.

### 3.2.4 `detect(reading: dict) -> dict` — Core Inference Function

Complete step-by-step execution trace:

**Step 1 — Extract machine_id:**
```python
machine_id = reading.get("machine_id", "PUMP-001")
```

**Step 2 — Load normalized scaler:**
```python
scaler = self._scaler_cache.get(machine_id)
if scaler is None:
    scaler = load_scaler(machine_id)
    self._scaler_cache[machine_id] = scaler
```

**Step 3 — Extract sensor values in correct column order:**
```python
sensor_values = [reading[col] for col in SENSOR_COLUMNS]
# = [temperature, motor_current, vibration, speed, pressure]
```

**Step 4 — Reshape and normalize:**
```python
X_raw = np.array(sensor_values).reshape(1, -1)   # shape: (1, 5)
X_norm = scaler.transform(X_raw)                  # Z-score transform
```

**Step 5 — Run Autoencoder reconstruction:**
```python
X_reconstructed = model.predict(X_norm, verbose=0)  # shape: (1, 5)
```

**Step 6 — Compute Mean Squared Error:**
```python
mse = np.mean((X_norm - X_reconstructed) ** 2)
# MSE measures how "surprised" the model is by this reading.
# Low MSE = normal, the model reconstructed it well.
# High MSE = anomaly, the model couldn't reconstruct it.
```

**Step 7 — Compute health score (0–100):**
```python
ANOMALY_THRESHOLD = 0.7187  # calibrated during training
health_score = max(0, 100 - (mse / ANOMALY_THRESHOLD) * 100)
```
A health score of 100 = perfectly normal; 0 = reconstruction error at 1× threshold; negative is clipped.

**Step 8 — Build and return result dict:**
```python
return {
    "is_anomaly":   mse > ANOMALY_THRESHOLD,
    "score":        float(mse),
    "threshold":    ANOMALY_THRESHOLD,
    "health_score": round(health_score, 1),
    "sensors":      reading,
    "machine_id":   machine_id,
}
```

### 3.2.5 Threshold Calibration

The threshold `0.7187` was determined empirically during model training by computing the 99th percentile MSE on the validation set of `state == "normal"` readings. This means 99% of genuinely normal readings fall below this threshold, giving a theoretical False Positive Rate of ~1%.

---

## 3.3 Autoencoder Architecture

The system supports two model architectures, trained and saved per-machine:

### 3.3.1 Dense Autoencoder (Primary)

```
Input (5 features)
    ↓
Dense(32, activation='relu')
    ↓
Dense(16, activation='relu')     ← Bottleneck: learns compressed "normal" representation
    ↓
Dense(32, activation='relu')
    ↓
Dense(5, activation='linear')    ← Reconstructed output
```

Training objective: minimize `Mean Squared Error` between input and reconstruction using only `state == "normal"` rows. The model learns to reconstruct normal patterns perfectly; when it encounters a fault pattern, reconstruction fails — producing high MSE.

### 3.3.2 LSTM Autoencoder (Sequential)

Used when temporal sequence matters:

```
Input (seq_len=10, features=5)
    ↓
LSTM(64, return_sequences=False)
    ↓
RepeatVector(seq_len)
    ↓
LSTM(64, return_sequences=True)
    ↓
TimeDistributed(Dense(5))
```

This architecture captures temporal dependencies — ideal for detecting sensor drift where the *trend* is anomalous but any single reading appears normal.

---

## 3.4 `AnomalyService` (`services/anomaly_service.py`)

### 3.4.1 Purpose & Design

The `AnomalyService` addresses a fundamental problem: **single-tick anomaly spikes are noise**. A single MSE exceedance could be caused by a transient electrical spike, EMI, or measurement error. The `AnomalyService` tracks *consecutive* anomaly counts to distinguish sustained faults from transient noise.

### 3.4.2 Constructor Parameters

```python
AnomalyService(
    on_anomaly: Callable[[dict], None] | None = None,
    consecutive_threshold: int = 3
)
```

- `on_anomaly`: callback function invoked when an escalated alert is generated. Allows callers (e.g., `MonitoringService`) to inject their own handler logic without subclassing.
- `consecutive_threshold`: number of continuous anomaly ticks before escalation. Default: 3 (meaning 3 seconds of sustained fault at 1Hz).

### 3.4.3 Internal State

```python
self._consecutive_count = 0     # tracks current run of anomaly ticks
self._total_processed   = 0     # lifetime total readings processed
self._total_anomalies   = 0     # lifetime total anomaly ticks
```

### 3.4.4 `process(reading: dict) -> dict`

Extended flow:

1. Calls `self._detector.detect(reading)` to get the MSE result.
2. Appends `timestamp` (UTC ISO-8601) to the result dict.
3. If `result["is_anomaly"]`:
   - Increments `_consecutive_count` and `_total_anomalies`
   - Sets `result["escalated"] = (consecutive_count >= threshold)`
   - Calls `format_alert(result)` from `alert_service.py` to build a structured alert dict
   - Calls `log_alert(alert)` to print + append to `data/processed/anomaly_alerts.jsonl`
   - If `self._on_anomaly` is set, fires the callback with the alert dict
4. If `result["is_anomaly"]` is False:
   - **Resets** `_consecutive_count = 0` (breaking the run)
   - Sets escalated = False
5. Returns augmented result dict

### 3.4.5 `stats` Property

```python
@property
def stats(self) -> dict:
    return {
        "total_processed":  self._total_processed,
        "total_anomalies":  self._total_anomalies,
        "anomaly_rate_pct": round(
            self._total_anomalies / max(self._total_processed, 1) * 100, 2
        ),
    }
```

Returns a rolling summary useful for system health dashboards.

---

## 3.5 `AlertService` (`services/alert_service.py`)

### 3.5.1 `format_alert(result: dict) -> dict`

Transforms the raw detector result into a human-readable, structured alert object:

```python
alert = {
    "severity":              "HIGH" if escalated else "WARNING",
    "timestamp":             UTC ISO-8601 string,
    "machine_id":            from sensors dict,
    "reconstruction_score":  rounded MSE to 6 decimal places,
    "threshold":             0.7187 rounded to 6 decimal places,
    "suspect_sensor":        result of _find_suspect_sensor(),
    "consecutive_anomalies": count of consecutive anomaly ticks,
    "sensor_readings":       all sensor values rounded to 3 decimal places,
}
```

**Severity Logic:** `"HIGH"` if the fault has been escalated (consecutive_count >= threshold). `"WARNING"` for first or second consecutive tick.

### 3.5.2 `_find_suspect_sensor(sensors: dict) -> str`

Heuristic to identify the most likely malfunctioning sensor:

For each sensor in `SENSOR_SCHEMA`:
1. Computes the **midpoint** of its normal range: `mid = (low + high) / 2`
2. Computes the **half-span**: `span = (high - low) / 2`
3. Computes **normalized deviation**: `dev = |value - mid| / span`
4. Returns the sensor with the highest deviation score

This is not a root-cause analysis — it's a probabilistic heuristic to direct the operator's initial attention.

### 3.5.3 `log_alert(alert: dict)`

Dual-channel logging:
1. **Console output**: ANSI-colored terminal print (red for HIGH, yellow for WARNING)
2. **JSONL file append**: Appends to `data/processed/anomaly_alerts.jsonl` — one JSON object per line. This log survives restarts and can be ingested for post-mortem analysis.

---

## 3.6 `MonitoringService` (`services/monitoring_service.py`)

### 3.6.1 Role

Acts as the **integration seam** between the detection layer and the agentic layer. It instantiates both `AnomalyService` and `OrchestratorAgent`, wiring them together via a callback.

### 3.6.2 `__init__`

```python
self.detector = AnomalyDetector()
self.orchestrator = OrchestratorAgent()
self.anomaly_tracker = AnomalyService(
    consecutive_threshold=3,
    on_anomaly=self._on_anomaly_confirmed
)
```

### 3.6.3 `_on_anomaly_confirmed(alert: dict)`

This callback is invoked by `AnomalyService` when an escalated fault is confirmed. It:
1. Maps alert severity to machine state: `"machine_fault"` or `"machine_warning"`
2. Constructs a structured context dict for the orchestrator:
   ```python
   {
     "machine_state":    "machine_fault" | "machine_warning",
     "anomaly_score":    float MSE,
     "suspect_sensor":   str,
     "recent_readings":  dict of sensor values,
   }
   ```
3. Calls `self.orchestrator.handle_anomaly(full_alert_data)` — this triggers the full LangGraph DAG (see Part 4)
4. Prints the full orchestrator result to console (for manual testing)

### 3.6.4 `process_reading(reading_dict, timestamp)`

The public interface. Simply delegates to `self.anomaly_tracker.process(reading_dict)`, which includes the entire callback chain. This is the single method called by `anomaly_routes.py`.

---

## 3.7 Anomaly Routes (`api/anomaly_routes.py`)

### 3.7.1 `POST /detect`

**Input:** Pydantic model `SensorReading` containing the 5 float fields.
**Processing:** Calls `monitor_service.process_reading()`, updates `LATEST_READING_STATE` (a module-level dict used as a simple in-memory cache).
**Output:** JSON with `is_anomaly`, `escalated`, `score`, `threshold`.

This is the endpoint that `stream_listener.py` POSTs to. It is also the passive receiver for the simulator's `requests.post()` call.

### 3.7.2 `GET /history?limit=20`

Connects to `NeonVectorStore().get_recent_events()` to retrieve persisted anomaly records from PostgreSQL. Serializes `datetime` objects to ISO strings before returning.

### 3.7.3 `GET /latest`

Returns the `LATEST_READING_STATE` module-level cache dict. This is a cheap polling endpoint for passive UI observers.

### 3.7.4 `LATEST_READING_STATE` Module Cache

```python
LATEST_READING_STATE = {
    "reading":   None,
    "is_anomaly": False,
    "escalated":  False,
    "score":      0.0,
    "threshold":  0.7187
}
```

Updated on every `/detect` call. This is **not thread-safe** in a multi-worker Uvicorn setup, but acceptable for a single-worker research deployment.

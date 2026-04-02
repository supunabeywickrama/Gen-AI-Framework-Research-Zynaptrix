# Part 2: Simulator, Ingestion & Preprocessing

## 2.1 The Anomaly Injector (`simulator/anomaly_injector.py`)

This module defines independent, pure functions that return sensor reading dictionaries for each possible machine state. Both the **real-time simulator** and the **batch dataset generator** import from this same module, guaranteeing that training data distributions match live simulation distributions — preventing train/test distribution shift.

### 2.1.1 `get_machine_config(machine_id: str) -> dict`
Returns the base Gaussian distribution parameters `(mean, std)` for each sensor per machine type, supporting multi-asset fleet simulation. Machine detection is string-based (case-insensitive `in` check):

| Machine Type | Temperature (°C) | Motor Current (A) | Vibration (mm/s) | Speed (RPM) | Pressure |
|---|---|---|---|---|---|
| **PUMP-001** (default) | μ=180, σ=2 | μ=4.5, σ=0.5 | μ=0.8, σ=0.1 | μ=160, σ=5 | μ=4.5, σ=0.2 |
| **LATHE-002** | μ=45, σ=2 | μ=12.5, σ=1.5 | μ=0.15, σ=0.05 | μ=3200, σ=20 | μ=8.5, σ=0.5 |
| **TURBINE-003** | μ=850, σ=15 | μ=450, σ=5 | μ=1.2, σ=0.2 | μ=15000, σ=50 | μ=32, σ=1 |

### 2.1.2 State Generator Functions

**`normal_reading(machine_id)`**
Generates stable Gaussian noise around each sensor's mean. Represents a healthy machine in steady-state operation. Uses `np.random.normal(mu, sigma)` for each sensor in the config.

**`machine_fault_reading(machine_id)`**
Simulates bearing wear or mechanical failure. Deviations applied to each sensor:
- `motor_current`: Multiplied by **1.6x** mean, **2x** std (high current draw from friction)
- `vibration`: Multiplied by **3.5x** mean, **5x** std (severe vibration from imbalance)
- `speed`: Reduced to **0.8x** mean, **2x** std (rotational slowing)
- `pressure`: Reduced to **0.9x** (loss of hydraulic/pneumatic efficiency)
- `temperature`: Slightly *decreased* (counter-intuitive: cooling from reduced friction before heat builds up)

**`sensor_freeze_reading(frozen_values)`**
Accepts a previously captured normal snapshot and returns *exactly* those values every tick. This simulates a stuck sensor where the readings appear suspiciously constant. The AnomalyDetector catches this via rolling-std analysis.

**`sensor_drift_reading(drift_step, machine_id)`**
Takes a normal reading and additively increases `temperature` by `drift_step` each tick. This creates a monotonically rising temperature curve that gradually exceeds the normal distribution, catching the Autoencoder's attention over time.

**`idle_reading()`**
Returns a machine-agnostic idle state: temperature ≈ 25°C (ambient), motor_current = 0.0, vibration = 0.0, speed = 0.0, pressure = 0.0. Models a powered-off state.

**`STATE_GENERATORS` dict:**
A dispatch table mapping state string names to their generator functions. Used for extensibility.

---

## 2.2 Real-Time Sensor Simulator (`simulator/sensor_simulator.py`)

### 2.2.1 Overview
The main simulator loop generates one reading per second (configurable via `interval_seconds`) and performs two actions simultaneously:
1. Writes the reading to InfluxDB via `InfluxWriter.write_sensor_reading()`
2. POSTs the reading to `{API_URL}/api/telemetry/push` to trigger WebSocket broadcasting to the frontend

### 2.2.2 `pick_state(current_state, drift_counter) -> str`
A probabilistic state selection function applied at the start of each state block:

| Next State | Probability |
|---|---|
| `normal` | 70% |
| `machine_fault` | 15% |
| `sensor_freeze` | 8% |
| `sensor_drift` | 4% |
| `idle` | 3% |

### 2.2.3 `simulate(machine_id, interval_seconds)` — Main Loop

State persistence logic ensures faults aren't just single-tick blips:
1. Calls `pick_state()` to select a new state
2. Chooses `state_duration = random.randint(3, 30)` — the fault persists for 3–30 ticks
3. Iterates `state_counter` each tick; transitions only when `state_counter >= state_duration`
4. For `sensor_freeze`: captures a `frozen_snapshot` at the first tick and returns it unchanged for all subsequent ticks in that state block
5. For `sensor_drift`: increments `drift_step` by `random.uniform(0.1, 0.5)` per tick

On each tick, the reading dict is augmented with `reading["machine_id"] = machine_id` before being written/pushed. This machine_id tag is critical for InfluxDB filtering and anomaly routing.

**CLI interface:**
```bash
python sensor_simulator.py --machine_id LATHE-002
```

---

## 2.3 Batch Dataset Generator (`generate_dataset.py`)

Used to create the CSV training data consumed by the Autoencoder training scripts.

### 2.3.1 `generate_dataset(machine_id, total_rows, seed) -> pd.DataFrame`

Steps:
1. Builds a list of `(state, count)` block tuples from `STATE_ROW_DISTRIBUTION` in `config/settings.py`. Each count is further broken into random sub-blocks of size 10–120 rows.
2. Shuffles the block list (so states appear in random order like real operations)
3. Iterates blocks, calls the appropriate injector function, and accumulates rows into a DataFrame
4. Applies Pandas `reset_index(drop=True)` and trims to exactly `total_rows`

**Output Schema:**
```
timestamp | machine_id | temperature | motor_current | vibration | speed | pressure | state
```

The `state` column is **dropped during Autoencoder training** — it's only used by the normalization scaler to fit on `state == "normal"` rows only (see Section 2.5).

### 2.3.2 `validate_dataset(df, machine_id)`
Prints a statistical summary including:
- Total rows and columns
- State distribution counts and percentages
- Per-sensor mean ± standard deviation

CLI usage:
```bash
python generate_dataset.py --machine_id PUMP-001 --rows 20000
```

---

## 2.4 InfluxDB Integration

### 2.4.1 `ingestion/influx_writer.py` — `InfluxWriter` Class

**Constructor:** Instantiates an `InfluxDBClient` using credentials from `config/influx_config.py` and creates a synchronous write API (`SYNCHRONOUS` mode ensures the write blocks until confirmed — no buffering).

**`write_sensor_reading(reading, state)`:**
Constructs an InfluxDB `Point` object using the line protocol:
- Tags: `machine_id` (string tag, indexed), `state` (current behavioral state tag)
- Fields: All 5 sensor columns (rounded to 3 decimal places)
- Timestamp: Defaults to server time (nanosecond precision)

Tags vs Fields distinction matters in InfluxDB: Tags are indexed for fast querying; Fields are the actual float measurements.

**`close()`:** Explicitly closes the client connection to prevent TCP connection leaks.

### 2.4.2 `ingestion/stream_listener.py` — InfluxDB Polling Daemon

This service runs as a separate backgroundprocess and polls InfluxDB every 1 second using a **Flux query**:

```flux
from(bucket: "sensors")
  |> range(start: -5s)
  |> filter(fn: (r) => r["_measurement"] == "sensor_readings")
  |> last()
```

This retrieves the most recent data point within the last 5-second window. The `last()` aggregation ensures idempotency — the same reading isn't forwarded twice (checked against `last_time`).

When a new de-duplicated reading is found containing all required `SENSOR_COLUMNS`, it is forwarded via HTTP POST to `/anomaly/detect` on the FastAPI server.

---

## 2.5 Preprocessing Pipeline

### 2.5.1 `preprocessing/data_cleaning.py`

Three independent utility functions, composable via the `clean()` pipeline function:

**`drop_missing(df, threshold=0.5)`**
Drops rows where more than 50% of sensor values are NaN. Fills remaining NaNs with `ffill()` (forward-fill from last valid reading) followed by `bfill()` (backward-fill for leading NaNs). This maintains temporal continuity.

**`clip_outliers(df, sigma=4.0)`**
Per-column outlier clipping: computes the population mean and std, then clips values to `[mean - 4σ, mean + 4σ]`. The 4-sigma threshold is deliberately high to preserve genuine fault signals while removing sensor hardware errors (e.g., -9999 placeholder readings).

**`detect_sensor_freeze(df, window=10, tolerance=1e-6)`**
Rolls a standard deviation window of 10 ticks across each sensor. Any sensor whose rolling std falls below `1e-6` (effectively zero variance) is flagged as frozen. Returns a boolean Series for masking / alerting.

### 2.5.2 `preprocessing/normalization.py`

**Critical Design Decision:** The `StandardScaler` is fitted **exclusively on `state == "normal"` rows**. This is intentional: fitting on all states including faults would compress the fault signal into the "normal" range, making anomalies invisible. By fitting only on normal data, fault readings will produce high Z-scores after transformation, making the Autoencoder's reconstruction error larger.

**Functions:**
- `fit_scaler(df)`: Filters for normal rows, fits StandardScaler
- `save_scaler(scaler, machine_id)`: Serializes scaler to `data/processed/scaler_{machine_id}.pkl`
- `load_scaler(machine_id)`: Loads the pickled scaler for inference time
- `normalize(df, scaler)`: Applies `scaler.transform()` to sensor columns only
- `fit_and_normalize(df)`: Convenience function combining fit + normalize

### 2.5.3 `preprocessing/feature_engineering.py`

Adds time-windowed features on top of the 5 raw sensor readings, creating a 15-feature input vector:

**`add_rolling_stats(df, window=10)`**
For each of the 5 sensors: adds `{col}_roll_mean` and `{col}_roll_std` columns. `min_periods=1` ensures no NaN at the start of the series. Rolling std is 0-filled if insufficient history.

**`add_delta_features(df)`**
For each of the 5 sensors: adds `{col}_delta` = first-order difference (`df[col].diff()`). Captures rate of change — critical for detecting rapid fault onset (e.g., a sudden 2A spike in motor current).

**`build_feature_matrix(df)`**
Full pipeline: applies rolling stats + delta features and returns a DataFrame of exactly 15 columns (5 raw + 5 roll_mean + 5 delta). The `_roll_std` columns can be enabled separately. This matrix is what the Autoencoder is trained on and receives for inference.

**Feature Column Order:**
```
[temperature, motor_current, vibration, speed, pressure,
 temperature_roll_mean, motor_current_roll_mean, vibration_roll_mean, speed_roll_mean, pressure_roll_mean,
 temperature_delta, motor_current_delta, vibration_delta, speed_delta, pressure_delta]
```

---

## 2.6 Configuration (`config/settings.py`)

The settings module centralizes all critical constants:

**`SENSOR_COLUMNS`**: `["temperature", "motor_current", "vibration", "speed", "pressure"]` — the ordered list of input features.

**`SENSOR_SCHEMA`**: A detailed dict with each sensor's `normal_range`, `unit`, and `description`. Used by `alert_service.py` to identify the most anomalous sensor by comparing readings to the midpoint of the normal range.

**`STATE_ROW_DISTRIBUTION`**: Controls the class balance in the generated training set:
```python
{
  "normal":        14000,  # 70% — ensures model knows "normal" well
  "machine_fault":  3000,  # 15%
  "sensor_freeze":  1500,  # 7.5%
  "sensor_drift":    900,  # 4.5%
  "idle":            600,  # 3%
}
```

**`MOCK_DATA_PATH`**: `data/mock_data/generated_sensor_data.csv`
**`PROCESSED_DATA_PATH`**: `data/processed/normalized_data.csv`

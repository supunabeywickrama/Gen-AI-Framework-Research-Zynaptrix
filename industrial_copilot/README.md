# 🏭 Industrial Copilot — Generative AI Multi-Agent System

> **An enterprise-grade AI-powered industrial monitoring and predictive maintenance platform.**  
> Real-time sensor telemetry, dynamic machine enrollment, anomaly detection, and guided repair copilot — all in one system.

---

## ✨ Today's Feature Updates (April 2, 2026)

### 🧠 Automated Dynamic Sensor & Datasheet Intelligence

#### 📄 AI-Powered PDF Datasheet Parsing
- **Richer extraction** — GPT-4o now extracts 7 parameters per sensor from uploaded datasheets:
  - `mu` / `sigma` — normal operating statistics
  - `min_normal` / `max_normal` — healthy operating band
  - `fault_high` / `fault_low` — realistic alarm thresholds from the spec sheet
  - `unit` — actual measurement unit (°C, A, rpm, bar, mm/s…)
  - `icon_type` — AI-chosen dashboard icon category (23 types supported)
- **No-PDF fallback** — when no datasheet is uploaded, GPT-4o estimates all parameters from the sensor name alone
- **Name-based heuristic fallback** — graceful degradation if OpenAI is unavailable

#### 🚨 Realistic Anomaly Generation
- `machine_fault_reading()` now pushes sensors toward their actual **fault thresholds** from the config, not arbitrary multipliers
- `idle_reading()` uses physics-aware values (temperature → ambient 25 °C, current/speed → 0)
- `sensor_drift_reading()` drifts incrementally toward the real fault boundary over time
- Built-in profiles for PUMP, LATHE, TURBINE with extended 6-tuple config format

#### 🔄 Automated Machine Onboarding Pipeline
- Registration → Dataset Generation → **Normalization (NEW)** → Model Training
- Normalization step was missing from the pipeline and has been added — AI models now train correctly every time
- All pipeline steps run asynchronously in the background after registration

---

### 📊 Dynamic Dashboard Enhancements

#### 🎯 Smart Sensor Icons (AI-Determined)
- Dashboard sensor cards now display **semantically correct icons** based on what each sensor measures
- 23 icon types mapped to Lucide React components:

| `icon_type`   | Icon         | Example Sensor               |
|---------------|--------------|------------------------------|
| `temperature` | Thermometer  | Thermistor, PT100, RTD       |
| `current`     | Zap          | LEM CT, Motor Current Sensor |
| `vibration`   | Vibrate      | Accelerometer, IMU           |
| `pressure`    | Gauge        | Pressure Transducer          |
| `speed`       | RotateCw     | Encoder, Tachometer          |
| `flow`        | Droplets     | Flow Meter                   |
| `voltage`     | Zap          | Voltage Divider              |
| `load`        | Weight       | Load Cell, Strain Gauge      |
| `power`       | TrendingUp   | Power Meter                  |
| `frequency`   | Radio        | Frequency Analyzer           |
| `gas`         | FlaskConical | Gas Detector, pH Sensor      |
| `generic`     | Server       | (fallback)                   |

#### 🔌 Sensor Config API
- New endpoint: `GET /api/machines/{machine_id}/config`
- Returns `sensors_meta[]` — sensor ID, human-readable name, icon type, and engineering unit
- Dashboard fetches config on machine select and shows sensor cards immediately — **even when the simulator is stopped**

#### 🃏 Enhanced Sensor Cards
- Each card now shows: **correct icon + accent color**, **human-readable sensor name**, **live value**, **real engineering unit**
- Cards derive from the registered config — all 4 TEA_0001 sensors now display correctly

---

## 🏗️ System Architecture

```
industrial_copilot/
├── backend/
│   ├── api/
│   │   ├── main_api.py           # FastAPI app, WebSocket, telemetry push, config endpoint
│   │   └── machine_api.py        # Machine registration, PDF parsing, pipeline trigger
│   ├── agents/
│   │   └── copilot_graph.py      # LangGraph multi-agent workflow
│   ├── services/
│   │   ├── datasheet_parser.py   # ✨ AI PDF parser (GPT-4o) with rich sensor extraction
│   │   └── anomaly_service.py    # Consecutive anomaly detection & alerting
│   ├── simulator/
│   │   ├── sensor_simulator.py   # Real-time sensor data simulator (per machine)
│   │   └── anomaly_injector.py   # ✨ Realistic fault injection using config thresholds
│   ├── models/
│   │   └── train_model.py        # Dense/LSTM autoencoder training per machine
│   ├── preprocessing/
│   │   └── normalization.py      # Dynamic sensor column detection & scaling
│   ├── generate_dataset.py       # Mock dataset generator (config-driven)
│   ├── scripts/
│   │   ├── migrate_db.py         # DB schema migration utility
│   │   └── enrich_tea_config.py  # ✨ Backfill rich config for existing machines
│   ├── unified_rag/              # Vector RAG pipeline for manual knowledge retrieval
│   ├── data/
│   │   └── processed/
│   │       └── sensor_configs.json  # ✨ Machine sensor registry (mu, sigma, ranges, icon)
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx          # ✨ Dynamic dashboard with smart icons & sensor meta
    │   │   └── machines/
    │   │       └── page.tsx      # Machine registry with dynamic sensor form + PDF upload
    │   └── store/
    │       └── slices/
    │           ├── copilotSlice.ts   # Anomaly state, chat history, procedure flow
    │           ├── machineSlice.ts   # ✨ SensorMeta[], fetchMachineConfig thunk
    │           └── simulatorSlice.ts # Start/stop simulator per machine
    └── package.json
```

---

## 🚀 Running the System

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL (with pgvector extension)
- InfluxDB 2.x
- OpenAI API Key

### Environment Setup
```bash
# Backend .env
DATABASE_URL=postgresql://user:pass@localhost:5432/industrial_copilot
OPENAI_API_KEY=sk-...
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=...
INFLUX_ORG=...
INFLUX_BUCKET=sensor_data
```

### Start Backend
```bash
cd industrial_copilot/backend
pip install -r requirements.txt
python -m uvicorn api.main_api:app --reload --host 0.0.0.0 --port 8000
```

### Start Frontend
```bash
cd industrial_copilot/frontend
npm install
npm run dev
# Open http://localhost:3000
```

### Run Database Migration (first time)
```bash
cd industrial_copilot/backend
python scripts/migrate_db.py
```

---

## 🔄 Machine Enrollment Flow

```
User fills form (Machine ID, Name, Sensors + PDF datasheets)
        │
        ▼
POST /api/machines  (multipart/form-data)
        │
        ▼
DatasheetParser.extract_sensor_config()
  ├── Reads PDF text (PyMuPDF)
  ├── GPT-4o extracts: mu, sigma, min/max, fault thresholds, unit, icon_type
  └── Saves to sensor_configs.json
        │
        ▼
Background Pipeline (async subprocess)
  1. generate_dataset.py   → creates 20,000 row mock dataset
  2. normalization.py      → scales sensor columns, saves scaler.pkl
  3. train_model.py        → trains Dense/LSTM autoencoder, saves model.keras
        │
        ▼
Dashboard automatically fetches new machine config
  └── Shows sensor cards with correct icons immediately
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/machines` | List all registered machines |
| `POST` | `/api/machines` | Register machine + trigger training |
| `GET`  | `/api/machines/{id}/config` | Sensor metadata (icon_type, unit, name) |
| `GET`  | `/api/machines/{id}/anomalies` | Anomaly history for machine |
| `POST` | `/api/telemetry/push` | Push live sensor reading → WebSocket broadcast |
| `WS`   | `/ws/telemetry` | WebSocket stream for live dashboard |
| `POST` | `/api/copilot/invoke` | Trigger AI copilot diagnosis |
| `POST` | `/api/simulator/start` | Start simulator for machine |
| `POST` | `/api/simulator/stop` | Stop simulator for machine |
| `GET`  | `/api/simulator/status` | List running simulators |

---

## 🤖 AI Agent Architecture

The copilot uses a **LangGraph multi-agent workflow**:

```
User Query
    │
    ▼
[Sensor Agent]    — interprets live telemetry context
    │
    ▼
[RAG Agent]       — retrieves relevant manual sections via pgvector
    │
    ▼
[Diagnostic Agent]— generates diagnosis with retrieved context
    │
    ▼
[Strategy Agent]  — creates step-by-step repair procedure (JSON structured)
    │
    ▼
[Critic Agent]    — reviews and improves the plan
    │
    ▼
Final Execution Plan → persisted to DB → streamed to UI
```

---

## 🛡️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Redux Toolkit, Recharts, Lucide Icons |
| Backend | FastAPI, Python 3.10, LangGraph, LangChain |
| AI | OpenAI GPT-4o (diagnosis, PDF parsing, procedure generation) |
| Vector DB | PostgreSQL + pgvector |
| Time-Series | InfluxDB 2.x |
| ML Models | TensorFlow/Keras (Dense + LSTM Autoencoders) |
| PDF Parsing | PyMuPDF (fitz) |
| Deployment | Docker Compose |

---

*Made with ❤️ for intelligent industrial operations.*

# 🏭 Industrial Copilot — Generative AI Multi-Agent System

> **An enterprise-grade AI-powered industrial monitoring and predictive maintenance platform.**  
> Real-time sensor telemetry, dynamic machine enrollment, anomaly detection, and guided repair copilot — all in one system.

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
python -m venv .venv (if not exists)
.\.venv\Scripts\Activate.ps1      
pip install -r requirements.txt
python -m uvicorn api.main_api:app --reload
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

*Made with Zynaptrix AI Research Team for intelligent industrial operations.*

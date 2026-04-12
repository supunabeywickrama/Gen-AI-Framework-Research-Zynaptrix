# 🏭 Industrial Copilot — Generative AI Multi-Agent System

> **An enterprise-grade AI-powered industrial monitoring and predictive maintenance platform.**  
> Real-time sensor telemetry, dynamic machine enrollment, anomaly detection, AI validation layer, and guided repair copilot — all in one system.

---


## 🏗️ System Architecture

```
industrial_copilot/
├── backend/
│   ├── api/
│   │   ├── main_api.py           # FastAPI app, WebSocket, telemetry push, AI validation callback
│   │   ├── machine_api.py        # Machine registration, PDF parsing, AI-validated feedback
│   │   ├── assistant_api.py      # Central Assistant API
│   │   └── copilot_chat_api.py   # Copilot chat & incident resolution
│   ├── agents/
│   │   ├── copilot_graph.py      # LangGraph multi-agent workflow with AI validation
│   │   ├── ai_automation_engineer.py  # ✨ NEW: AI Automation Engineer Agent
│   │   └── validation_prompts.py      # ✨ NEW: GPT-4o validation prompts with few-shot examples
│   ├── services/
│   │   ├── datasheet_parser.py   # ✨ AI PDF parser (GPT-4o) with AI validation
│   │   ├── anomaly_service.py    # Consecutive anomaly detection, TemporalAnalyzer, hybrid confidence
│   │   └── sensor_config_loader.py    # ✨ NEW: Physics-based validation singleton
│   ├── simulator/
│   │   ├── sensor_simulator.py   # Real-time sensor data simulator (per machine)
│   │   └── anomaly_injector.py   # ✨ Realistic fault injection using config thresholds
│   ├── models/
│   │   └── train_model.py        # Dense/LSTM autoencoder training per machine
│   ├── preprocessing/
│   │   └── normalization.py      # Dynamic sensor column detection & scaling
│   ├── generate_dataset.py       # ✨ AI-enhanced dataset generator with anomaly patterns
│   ├── scripts/
│   │   ├── migrate_db.py         # DB schema migration utility
│   │   ├── add_ai_validation_columns.py  # ✨ NEW: AI validation DB migration
│   │   └── enrich_tea_config.py  # Backfill rich config for existing machines
│   ├── unified_rag/              # Vector RAG pipeline for manual knowledge retrieval
│   │   └── db/
│   │       └── models.py         # ✨ Updated: AnomalyRecord with AI validation fields
│   ├── data/
│   │   └── processed/
│   │       ├── sensor_configs.json       # Machine sensor registry
│   │       └── anomaly_patterns_*.json   # ✨ NEW: AI-generated anomaly patterns
│   └── requirements.txt
│
└── frontend/
    ├── src/
    │   ├── app/
    │   │   ├── page.tsx          # ✨ Dashboard with AI validation status & PDF export
    │   │   └── machines/
    │   │       └── page.tsx      # Machine registry with dynamic sensor form + PDF upload
    │   ├── components/
    │   │   └── ExportProgressModal.tsx  # ✨ NEW: PDF export progress UI
    │   ├── professionalReportService.ts # ✨ NEW: jsPDF report generator
    │   └── store/
    │       └── slices/
    │           ├── copilotSlice.ts   # ✨ Updated: PDF export state & thunk
    │           ├── machineSlice.ts   # SensorMeta[], fetchMachineConfig thunk
    │           └── simulatorSlice.ts # Start/stop simulator per machine
    └── package.json
```

---

## ✨ New Features (April 2026)

### 📄 Professional PDF Export
Export diagnostic conversations as professional maintenance reports:

```
┌─────────────────────────────────────────────────────────────┐
│  DIAGNOSTIC REPORT                              ZYNAPTRIX   │
├─────────────────────────────────────────────────────────────┤
│  Report ID: #123    Machine: PUMP-001    Status: COMPLETE   │
├─────────────────────────────────────────────────────────────┤
│  ▌ PROBLEM DESCRIPTION                                      │
│    AI-extracted problem summary from conversation           │
├─────────────────────────────────────────────────────────────┤
│  ▌ DIAGNOSIS                                                │
│    Root cause analysis with technical details               │
├─────────────────────────────────────────────────────────────┤
│  ▌ SOLUTION / REPAIR PROCEDURE                              │
│    ① Step 1: Disconnect power                              │
│    ② Step 2: Inspect components                            │
│    ③ Step 3: Replace faulty parts                          │
├─────────────────────────────────────────────────────────────┤
│  ▌ REFERENCE DIAGRAMS                                       │
│    [Images from manual pages]                               │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- AI-powered content extraction using GPT-4o-mini
- Professional Zynaptrix branding and watermark
- Structured sections: Problem, Diagnosis, Solution Steps, Images
- Auto-download with timestamped filename
- Progress modal during generation

See [`Docs/PDF_EXPORT_FEATURE.md`](../Docs/PDF_EXPORT_FEATURE.md) for detailed documentation.

### 🧠 AI Validation Layer
High-accuracy anomaly classification using a 4-stage validation pipeline:

```
┌────────────────────────────────────────────────────────────────┐
│                    VALIDATION PIPELINE                          │
├────────────────────────────────────────────────────────────────┤
│  Stage 1: Physics Violations Check                              │
│  └── Check readings against physical limits (impossible values) │
│                                                                 │
│  Stage 2: Temporal Pattern Analysis                             │
│  └── Detect sudden spikes vs sustained anomalies                │
│                                                                 │
│  Stage 3: Cross-Sensor Correlation                              │
│  └── Validate physical plausibility across sensors              │
│                                                                 │
│  Stage 4: AI High-Accuracy Classification                       │
│  └── Multi-hypothesis analysis with root cause identification  │
└────────────────────────────────────────────────────────────────┘
```

**Classifications:**
- `TRUE_FAULT` — Confirmed equipment failure requiring action
- `SENSOR_GLITCH` — Transient sensor anomaly, no action needed
- `NORMAL_WEAR` — Expected degradation, schedule maintenance

### 🤖 AI Automation Engineer Agent
New agent providing engineering expertise:
- `validate_sensor_config()` — Validates extracted sensor configurations
- `generate_anomaly_patterns()` — Creates realistic fault patterns for training
- `cross_validate_sensors()` — Checks sensor relationship consistency
- `high_accuracy_fault_classification()` — Multi-hypothesis fault analysis

### ✅ AI-Validated Feedback System
When operators archive incidents:
1. AI validates feedback quality and technical relevance
2. Returns suggestions if feedback needs improvement
3. Generates personalized thank you message with Central Assistant link
4. Persists validated results to database for future RAG retrieval

### 📊 Autoencoder Evaluation Pipeline
Comprehensive evaluation metrics pipeline for the anomaly detection model (Dense & LSTM):
- Computes 8 key metrics (Precision, Recall, F1, AUC-ROC, FPR, FNR, Threshold, MSE Separation).
- Automatically evaluates new machines directly after ML training during enrollment.
- Generates 4 publication-quality visualizations per machine (ROC Curve, Confusion Matrix, MSE Distribution, Threshold Sweep).
- Accessible via `/api/evaluation/` endpoints.
- See `docs/evaluation_metrics_implementation.md` for full details.

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
python -m venv .venv  # (if not exists)
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

# NEW: Add AI validation columns
python scripts/add_ai_validation_columns.py
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
  ├── ✨ AI Automation Engineer validates extracted config
  └── Saves to sensor_configs.json
        │
        ▼
Background Pipeline (async subprocess)
  1. generate_dataset.py   → creates 20,000 row mock dataset
  2. ✨ AI generates realistic anomaly patterns from sensor specs
  3. normalization.py      → scales sensor columns, saves scaler.pkl
  4. train_model.py        → trains Dense/LSTM autoencoder, saves model.keras
        │
        ▼
Dashboard automatically fetches new machine config
  └── Shows sensor cards with correct icons immediately
```

---

## 🔍 Anomaly Detection & Validation Flow

```
Live Sensor Data → Anomaly Service
        │
        ▼
ML Model detects anomaly (MSE threshold exceeded)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│               LANGGRAPH VALIDATION PIPELINE               │
├───────────────────────────────────────────────────────────┤
│  SensorStatus → ValidationEngineer → Diagnostic           │
│                      │                    │                │
│                      ▼                    ▼                │
│               4-Stage AI Analysis    DB Persistence        │
│               (Physics + Temporal    (ai_validation_status │
│                + Correlation +        fault_category,      │
│                AI Classification)     confidence_score)    │
│                      │                                     │
│                      ▼                                     │
│  KnowledgeRetrieval → Strategy → Critic → Final Output    │
└───────────────────────────────────────────────────────────┘
        │
        ▼
WebSocket broadcast to Dashboard
  └── Shows validation status (TRUE_FAULT / SENSOR_GLITCH / NORMAL_WEAR)
```

---

## 📡 Key API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/machines` | List all registered machines |
| `POST` | `/api/machines` | Register machine + AI validation + trigger training |
| `GET`  | `/api/machines/{id}/config` | Sensor metadata (icon_type, unit, name) |
| `GET`  | `/api/machines/{id}/anomalies` | Anomaly history with AI validation status |
| `POST` | `/api/telemetry/push` | Push live sensor reading → WebSocket broadcast |
| `WS`   | `/ws/telemetry` | WebSocket stream for live dashboard |
| `POST` | `/api/copilot/invoke` | Trigger AI copilot diagnosis |
| `POST` | `/api/chat-history/{id}/resolve` | ✨ Archive incident with AI-validated feedback |
| `GET`  | `/api/assistant/sessions/{id}/report` | ✨ Generate PDF report data |
| `POST` | `/api/simulator/start` | Start simulator for machine |
| `POST` | `/api/simulator/stop` | Stop simulator for machine |
| `GET`  | `/api/simulator/status` | List running simulators |
| `POST` | `/api/assistant/chat` | Central Assistant chat |

---

## 🤖 AI Agent Architecture

The copilot uses a **LangGraph multi-agent workflow** with AI validation:

```
Anomaly Detected / User Query
        │
        ▼
[Sensor Agent]         — interprets live telemetry context
        │
        ▼
[Validation Engineer]  — ✨ NEW: 4-stage AI validation
        │                  (Physics + Temporal + Correlation + AI)
        ▼
[Diagnostic Agent]     — generates diagnosis with AI validation results
        │                  Persists to DB after validation
        ▼
[RAG Agent]            — retrieves relevant manual sections via pgvector
        │
        ▼
[Strategy Agent]       — creates step-by-step repair procedure (JSON structured)
        │
        ▼
[Critic Agent]         — reviews and improves the plan
        │
        ▼
Final Execution Plan → streamed to UI with validation status
```

---

## 🗄️ Database Schema

### AnomalyRecord (Updated)
```sql
CREATE TABLE anomaly_records (
    id SERIAL PRIMARY KEY,
    machine_id VARCHAR(50),
    timestamp TIMESTAMP,
    type VARCHAR(50),
    score FLOAT,
    sensor_data JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    -- ✨ NEW: AI Validation Fields
    ai_validation_status VARCHAR(50),    -- TRUE_FAULT, SENSOR_GLITCH, NORMAL_WEAR
    fault_category VARCHAR(50),          -- mechanical, thermal, electrical, process, sensor
    ai_confidence_score FLOAT,           -- 0.0 to 1.0
    ai_engineering_notes TEXT            -- Detailed AI analysis
);
```

---

## 🛡️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Redux Toolkit, Recharts, Lucide Icons |
| Backend | FastAPI, Python 3.10, LangGraph, LangChain |
| AI | OpenAI GPT-4o (diagnosis, PDF parsing, validation, procedure generation) |
| AI Agents | AI Automation Engineer, Validation Engineer, Strategy Agent |
| Vector DB | PostgreSQL + pgvector |
| Time-Series | InfluxDB 2.x |
| ML Models | TensorFlow/Keras (Dense + LSTM Autoencoders) |
| PDF Parsing | PyMuPDF (fitz) |
| Deployment | Docker Compose |

---

## 📋 Changelog

### April 7, 2026
- ✨ **NEW: Professional PDF Export** for Central Assistant conversations
  - AI-powered content extraction (problem, diagnosis, steps)
  - Zynaptrix branding with watermark
  - Reference diagram embedding
  - Progress modal during generation
- ✨ Added AI Validation Layer with 4-stage pipeline
- ✨ Added AI Automation Engineer Agent
- ✨ Added AI-validated feedback system for incident archival
- ✨ Added thank you message with Central Assistant link
- ✨ Added database columns for AI validation results
- ✨ Enhanced diagnostic node with DB persistence
- ✨ Added physics-based sensor config validation

See `CHANGELOG-2026-04-07.md` for detailed implementation notes.
See `Docs/PDF_EXPORT_FEATURE.md` for PDF export documentation.

---

*Made with ❤️ by Zynaptrix AI Research Team for intelligent industrial operations.*

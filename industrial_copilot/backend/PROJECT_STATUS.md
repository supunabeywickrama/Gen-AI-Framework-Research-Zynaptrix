# 🏭 Industrial AI Copilot — Project Status

> **Tea Bag Packing Machine — Predictive Maintenance & Anomaly Detection System**
> Last updated: 2026-03-16

---

## 🗺️ Project Architecture Overview

```
Sensor Simulator
     │
     ▼
InfluxDB (time-series) ──► Preprocessing ──► Anomaly Detection Model
                                                      │
                                                      ▼
Neon PostgreSQL (pgvector) ◄─── Knowledge Docs     Alerts / JSONL Log
     │                                                │
     ▼                                                ▼
Knowledge Agent                              Multi-Agent Orchestrator
     └──────────────────────────────────────────────►│
                                                      ▼
                                               FastAPI REST API
                                                      │
                                                      ▼
                                          Streamlit Live Dashboard
```

---

## ✅ Phase 1 — Sensor Modeling & Mock Dataset (COMPLETE)

### What Was Built
| File | Description |
|---|---|
| `config/settings.py` | Sensor schema, 5 machine states, row distribution, data paths |
| `config/influx_config.py` | InfluxDB connection (reads from `.env`) |
| `config/neon_config.py` | Neon PostgreSQL connection (reads from `.env`) |
| `simulator/anomaly_injector.py` | State-based sensor reading generators for all 5 states |
| `simulator/sensor_simulator.py` | Real-time simulator — streams readings to InfluxDB every second |
| `generate_dataset.py` | Block-based 20,000-row dataset generator |
| `preprocessing/data_cleaning.py` | Missing value handling, outlier clipping, freeze detection |
| `preprocessing/normalization.py` | StandardScaler fitted on normal-state data only |
| `preprocessing/feature_engineering.py` | Rolling mean/std and Δ-delta features |
| `validate_dataset.py` | Dataset quality validation (3 charts generated) |

### Dataset Statistics
| State | Rows | % |
|---|---|---|
| normal | 14,000 | 70% |
| machine_fault | 3,000 | 15% |
| sensor_freeze | 1,500 | 7.5% |
| sensor_drift | 1,000 | 5% |
| idle | 500 | 2.5% |

**Output:** `data/mock_data/generated_sensor_data.csv` (20,000 rows)

---

## ✅ Phase 2 — Anomaly Detection Model (COMPLETE)

### What Was Built
| File | Description |
|---|---|
| `models/autoencoder_model.py` | Dense autoencoder (5→16→8→4→8→16→5) with dropout |
| `models/lstm_autoencoder.py` | LSTM autoencoder for sequence-based detection |
| `models/train_model.py` | Training pipeline — trained on normal data only |
| `models/detect_anomaly.py` | `AnomalyDetector` class — single-reading & batch inference |
| `services/anomaly_service.py` | Stateful service with consecutive-anomaly tracking + callback |
| `services/alert_service.py` | Alert formatter — identifies suspect sensor, logs to JSONL |
| `tests/test_model.py` | Unit tests: load, normal/fault/idle scoring, false positive rate |

### Model Performance
**Model:** Dense Autoencoder | **Threshold:** `0.7187` (mean + 2σ of training errors)

| State | Detection Rate | Avg Score |
|---|---|---|
| machine_fault | **100% ✅** | 10.81 |
| idle | **100% ✅** | 1360.90 |
| sensor_drift | 27.3% 🔶 | 0.55 |
| sensor_freeze | 7.2% 🔶 | 0.21 |
| normal (false positive) | 5.1% ⚡ | 0.22 |

> **Note:** Sensor freeze/drift are subtle by design. The LSTM model (already scaffolded) can improve these with temporal pattern detection.

### Model Artifacts Saved
- `data/processed/autoencoder.keras` — trained model
- `data/processed/scaler.pkl` — fitted normalizer
- `data/processed/thresholds.json` — detection threshold
- `data/processed/generated_sensor_data_anomaly_results.csv` — full batch results

---

## ✅ Phase 3 — Neon Vector DB + Knowledge Retrieval (COMPLETE)

### 3A — Neon PostgreSQL + pgvector Setup
- [x] `database/schema.sql` — pgvector table schema (`machine_documents` + `anomaly_events` + `active_alerts` view)
- [x] `database/neon_vector_store.py` — Insert, cosine similarity search, anomaly event logging

### 3B — Vector Pipeline (Document Embedding)
- [x] `vector_pipeline/document_parser.py` — 7 built-in maintenance docs + file parser with overlap chunking
- [x] `vector_pipeline/embedding_generator.py` — OpenAI text-embedding-3-small (768-dim) + sentence-transformers fallback
- [x] `vector_pipeline/vector_uploader.py` — CLI pipeline: parse → embed → upload to Neon (--clear, --status flags)

### 3C — Knowledge Agent (RAG)
- [x] `agents/knowledge_agent.py` — RAG: embed query → Neon similarity search → Gemini/GPT-4o synthesis → maintenance advice

---

## ✅ Phase 4 — Multi-Agent Orchestration (COMPLETE)

- [x] `agents/machine_health_agent.py` — Queries InfluxDB, summarises machine health status
- [x] `agents/sensor_status_agent.py` — Identifies which sensors are anomalous based on ranges
- [x] `agents/strategy_agent.py` — Recommends maintenance action based on fault type/severity
- [x] `agents/orchestrator_agent.py` — Boss Agent: coordinates all agents, retrieves RAG advice, logs event to Neon
- [x] `services/monitoring_service.py` — Continuous monitoring loop linking AnomalyDetector + Orchestrator

---

## ✅ Phase 5 — FastAPI Backend (COMPLETE)

- [x] `api/main_api.py` — FastAPI app entry point
- [x] `api/anomaly_routes.py` — `/anomaly/detect`, `/anomaly/history` endpoints
- [x] `api/health_routes.py` — `/health/status`, `/health/sensor/{id}` endpoints

---

## ✅ Phase 6 — Streamlit Live Dashboard (COMPLETE)

- [x] `dashboard/streamlit_dashboard.py` — Real-time sensor charts, anomaly feed, agent chat UI
- [x] `dashboard/graphs.py` — Plotly graph components (time-series, score gauges, heatmaps)

---

## ✅ Phase 7 — InfluxDB Streaming Integration (COMPLETE)

- [x] `ingestion/influx_writer.py` — Write live sensor readings to InfluxDB
- [x] `ingestion/stream_listener.py` — Subscribe to live stream, trigger inference on new readings
- [ ] `ingestion/mqtt_producer.py` — MQTT bridge (optional — skipped in favor of direct Influx pipeline)

---

## ✅ Phase 8 — Docker Deployment (COMPLETE)

- [x] Create `Dockerfile` for minimal Python execution
- [x] Create `docker-compose.yml` orchestrating 5 isolated containers
- [x] Implement dynamic `.env` internal container network routing

---

## 📋 Recommended Build Order

```
Phase 3A (Neon DB schema)
    → Phase 3B (Embed docs)
        → Phase 3C (Knowledge Agent)
            → Phase 4 (Multi-Agent Orchestration)
                → Phase 5 (FastAPI)
                    → Phase 6 (Dashboard)
                        → Phase 7 (InfluxDB Live Streaming)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Time-series DB | InfluxDB |
| Vector DB | Neon PostgreSQL + pgvector |
| ML Models | TensorFlow/Keras (Dense + LSTM Autoencoder) |
| Agent Framework | Custom Python agents |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Messaging | MQTT (optional) |
| Config | python-dotenv `.env` |

---

## 📁 Project Structure

```
industrial-ai-copilot/
├── config/              ✅ Settings, InfluxDB, Neon configs
├── simulator/           ✅ Anomaly injector + real-time simulator
├── preprocessing/       ✅ Cleaning, normalization, feature engineering
├── models/              ✅ Autoencoder + LSTM + training + inference
├── services/            ✅ Anomaly service, alert service, monitoring_service
├── vector_pipeline/     ✅ Document parser, embedder (OpenAI/local), uploader CLI
├── database/            ✅ Neon vector store + schema (pgvector)
├── agents/              ✅ knowledge, health, sensor, strategy, orchestrator
├── api/                 ✅ FastAPI routes + app entry
├── dashboard/           ✅ Streamlit UI + graphs
├── ingestion/           ✅ InfluxDB writer + stream listener
├── docker-compose.yml   ✅ Orchestrates full 5-container architecture
├── Dockerfile           ✅ Unified Python 3.10 microservice builder
├── data/                ✅ mock_data/ + processed/
├── tests/               ✅ test_model.py
├── generate_dataset.py  ✅
├── validate_dataset.py  ✅
├── requirements.txt     ✅
└── .env                 ✅
```

# 🚀 Getting Started — Industrial AI Copilot

> Predictive Maintenance System for Tea Bag Packing Machine

---

## 📋 Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| InfluxDB | 2.x | Time-series sensor storage |
| Neon PostgreSQL | (cloud) | Vector knowledge base |
| OpenAI or Gemini API key | — | LLM synthesis (Phase 3+) |

---

## ⚙️ 1. Clone & Create Virtual Environment

```bash
# Navigate to project folder
cd industrial-ai-copilot

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

---

## 📦 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` (~420 MB model) downloads on first use if no OpenAI key is set.

---

## 🔑 3. Configure Environment Variables

Open `.env` and fill in your credentials:

```env
# ── InfluxDB ──────────────────────────────────
INFLUX_URL=http://localhost:8086
INFLUX_TOKEN=your_influxdb_token
INFLUX_ORG=factory
INFLUX_BUCKET=machine_sensors

# ── Neon PostgreSQL + pgvector ────────────────
NEON_DB_URL=postgresql://user:password@host.neon.tech/dbname

# ── AI / LLM ──────────────────────────────────
GOOGLE_API_KEY=your_google_gemini_key
OPENAI_API_KEY=your_openai_key        # (optional — for embeddings + GPT-4o)
```

> The system uses **Gemini** for LLM synthesis and **OpenAI text-embedding-3-small** for embeddings.  
> If no OpenAI key, it falls back to local `sentence-transformers` automatically.

---

## 🗄️ 4. Set Up Neon Database Schema

Run the schema once on your Neon database to create tables and the pgvector index:

```bash
# Using psql (install from postgresql.org)
psql $NEON_DB_URL -f database/schema.sql

# Or using the connection string directly:
psql "postgresql://user:password@host.neon.tech/dbname" -f database/schema.sql
```

This creates:
- `machine_documents` table (pgvector knowledge base)
- `anomaly_events` table (alert log)
- `active_alerts` view

---

## 📚 5. Index the Knowledge Base (Phase 3)

Upload built-in maintenance documentation to Neon pgvector:

```bash
# Index built-in docs (7 maintenance documents, ~30 chunks)
python -m vector_pipeline.vector_uploader

# Force re-index (clears existing, re-uploads all)
python -m vector_pipeline.vector_uploader --clear

# Check current status
python -m vector_pipeline.vector_uploader --status
```

---

## 🤖 6. Test the Knowledge Agent

Run a quick RAG query to verify the full pipeline works:

```python
# In Python REPL or a test script
from agents.knowledge_agent import KnowledgeAgent

agent = KnowledgeAgent(top_k=3)
result = agent.query("Motor current is 7.5A and vibration is very high")

print(result["answer"])
print("\nSources:")
for s in result["sources"]:
    print(f"  - {s['title']} (score: {s['score']})")

agent.close()
```

---

## 📊 7. Generate / Validate the Mock Dataset

```bash
# Generate 20,000-row sensor dataset
python generate_dataset.py

# Validate and generate 3 quality charts
python validate_dataset.py
```

---

## 🧠 8. Train the Anomaly Detection Model

```bash
python -m models.train_model
```

Trained artifacts saved to `data/processed/`:
- `autoencoder.keras`
- `scaler.pkl`
- `thresholds.json`

---

## 🧪 9. Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📡 10. Run the Real-Time Sensor Simulator

Streams one reading per second to InfluxDB (requires InfluxDB running):

```bash
python -m simulator.sensor_simulator
```

---

## 🗺️ Run Order Summary

```
(One-time setup)
1. pip install -r requirements.txt
2. Configure .env
3. psql $NEON_DB_URL -f database/schema.sql
4. python -m vector_pipeline.vector_uploader

(Development)
5. python generate_dataset.py
6. python -m models.train_model
7. python -m pytest tests/

(Runtime — once Phase 4-6 is built)
8. uvicorn api.main_api:app --reload       ← FastAPI backend
9. streamlit run dashboard/streamlit_dashboard.py  ← Dashboard
10. python -m simulator.sensor_simulator   ← Live sensor stream
```

---

## 📁 Project Layout

```
industrial-ai-copilot/
├── .env                    ← Your credentials (never commit this)
├── requirements.txt
├── generate_dataset.py     ← Run to regenerate mock data
├── validate_dataset.py     ← Run to verify dataset quality
│
├── config/                 ← Centralized settings & DB configs
├── simulator/              ← Sensor simulator + anomaly injector
├── preprocessing/          ← Data cleaning, normalization, features
├── models/                 ← Autoencoder + LSTM + training pipeline
├── services/               ← Anomaly service, alert service
│
├── database/               ← Neon pgvector CRUD layer + SQL schema
├── vector_pipeline/        ← Doc parser → embedder → uploader
├── agents/                 ← Knowledge agent + multi-agent orchestrator
│
├── api/                    ← FastAPI endpoints (Phase 5)
├── dashboard/              ← Streamlit UI (Phase 6)
├── ingestion/              ← InfluxDB writer + MQTT (Phase 7)
│
├── data/
│   ├── mock_data/          ← generated_sensor_data.csv
│   └── processed/          ← Trained model artifacts
└── tests/                  ← Unit tests
```

---

## ⚠️ Common Issues

| Problem | Fix |
|---|---|
| `psycopg2` install fails on Windows | Use `psycopg2-binary` (already in requirements.txt) |
| `pgvector` extension missing on Neon | Run `CREATE EXTENSION vector;` in Neon SQL editor |
| `sentence-transformers` slow first run | Normal — downloads ~420 MB model once |
| InfluxDB connection refused | Ensure Docker/InfluxDB is running on port 8086 |
| OpenAI 429 rate limit | Embedder auto-retries 3× with exponential backoff |



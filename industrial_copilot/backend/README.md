# 🏭 Industrial AI Copilot

> **Predictive Maintenance System for Tea Bag Packing Machines**  
> An end-to-end framework for anomaly detection, root cause analysis, and maintenance strategy generation using GenAI and Time-Series data.

---

## 🛠️ Tech Stack & Architecture

- **Backend**: FastAPI (Python 3.10+)
- **Frontend**: Streamlit Dashboard
- **Database**: InfluxDB (Time-series) & Neon PostgreSQL (pgvector for Knowledge Base)
- **AI/LLM**: Google Gemini (Synthesis) & OpenAI / Sentence-Transformers (Embeddings)
- **Analytics**: TensorFlow/Keras (Autoencoder for Anomaly Detection)

---

## 📖 System Manual Running Procedure

To run the full Industrial AI Copilot system, ensure your environment is set up (see [GETTING_STARTED.md](GETTING_STARTED.md)) and follow these steps. It is recommended to use three separate terminal windows/slots:

### 1️⃣ Start the Backend API
The FastAPI backend handles anomaly routing and agent orchestration.
```bash
# In Terminal 1
cd industrial-ai-copilot
uvicorn api.main_api:app --reload
```
*Access API Docs at: http://localhost:8000/docs*

### 2️⃣ Start the Dashboard
The Streamlit interface provides live visualizations and the AI Copilot chat.
```bash
# In Terminal 2
cd industrial-ai-copilot
streamlit run dashboard/streamlit_dashboard.py
```
*Access Dashboard at: http://localhost:8501*

### 3️⃣ Start the Sensor Simulator
Streams real-time sensor data into InfluxDB for the system to analyze.
```bash
# In Terminal 3
cd industrial-ai-copilot
python -m simulator.sensor_simulator
```

---

## 🚀 Quick Setup Guide

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Environment Variables**: Configure `.env` (see `GETTING_STARTED.md` for template).
3. **Initialize DB**: `psql $NEON_DB_URL -f database/schema.sql`
4. **Prepare Data**:
   - `python generate_dataset.py` (Create mock data)
   - `python -m models.train_model` (Train anomaly model)
   - `python -m vector_pipeline.vector_uploader` (Index knowledge base)

For the full detailed setup, please refer to:  
👉 **[GETTING_STARTED.md](GETTING_STARTED.md)**

---

## 📂 Project Structure

```bash
industrial-ai-copilot/
├── agents/             # Knowledge, Health, and Strategy agents
├── api/                # FastAPI endpoints
├── dashboard/          # Streamlit UI components
├── database/           # PostgreSQL/pgvector schema & CRUD
├── models/             # ML training & inference logic
├── simulator/          # Live sensor data streaming
└── vector_pipeline/    # Documentation indexing (RAG)
```

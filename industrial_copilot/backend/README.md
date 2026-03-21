# 🏭 Industrial AI Copilot — Backend

> **Predictive Maintenance System for Industrial Packing Machines**  
> A research-grade framework for real-time anomaly detection, multimodal RAG retrieval, and LLM-based strategy orchestration.

---

## 🛠️ Tech Stack & Architecture

- **Core**: FastAPI (Python 3.10+)
- **Frontend**: Next.js 14+ (Dashboard)
- **Database**: Neon PostgreSQL (with `pgvector` for context)
- **Unified RAG**: 
  - **Layout Detection**: YOLOv8-DocLayout
  - **Vision**: OpenAI GPT-4o for technical diagram captioning
  - **Embeddings**: HuggingFace CLIP (Multimodal)
- **Analytics**: TensorFlow/Keras (Dense Autoencoder for Anomaly Detection)
- **Orchestration**: LangGraph (Multi-agent coordination)

---

## 📖 Running the System

Ensure your backend environment is active (using `.venv`) and follow these steps:

### 1. Start the Backend API
The FastAPI backend handles anomaly detection, telemetry streaming, and RAG orchestration.
```bash
# From the industrial_copilot/backend folder
.venv\Scripts\python -m uvicorn api.main_api:app --host 0.0.0.0 --port 8500 --reload
```
*Access API Docs at: http://localhost:8500/docs*

### 2. Pre-Running Checklist
Before using the full system, ensure these artifacts are generated:
- **Mock Data**: `python generate_dataset.py`
- **Normalization**: `python -m preprocessing.normalization`
- **Train Anomaly Model**: `python models/train_model.py` (Creates `autoencoder.keras`)
- **Ingest PDF Manuals**: Use the `/ingest-manual` endpoint or the Frontend Dashboard.

---

## 📂 Project Structure

```bash
industrial_copilot/backend/
├── agents/             # LangGraph orchestrators & tool-specific agents
├── api/                # FastAPI routes (Telemetry, RAG, Anomaly)
├── config/             # Environment & Sensor schema settings
├── data/               # Persistent storage for manuals and models
├── models/             # Autoencoder logic & YOLOv8 weights
├── preprocessing/      # Data standardization pipelines
├── services/           # Anomaly tracking & Alert logic
└── unified_rag/        # Core Multimodal RAG Engine (Parser, Embedder, Retriever)
```

---

## 🚀 Unified RAG Capabilities
The `unified_rag` module supports true multimodal retrieval:
1. **Layout Parsing**: Automatically separates text, tables, and figures from technical PDFs.
2. **Visual Intelligence**: Technical diagrams are interpreted by GPT-4o Vision to generate searchable captions.
3. **Multimodal Search**: Users can search for specific spare parts or machine settings, and the system can retrieve the exact diagram from the manual.

---
*Developed for Zynaptrix Industrial Research*

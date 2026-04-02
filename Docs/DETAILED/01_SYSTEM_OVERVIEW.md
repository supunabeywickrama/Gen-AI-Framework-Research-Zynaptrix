# Part 1: System Overview & Project Structure

## 1.1 What Is the Zynaptrix Industrial Copilot?

The Zynaptrix Industrial Copilot is a *production-ready agentic AI framework* for multi-asset industrial environments. It is not a simple dashboard or rule-based alerting tool. It is a fully automated diagnostic system that:

1. **Continuously monitors** high-frequency sensor streams from industrial machines (Pumps, Lathes, Turbines).
2. **Detects anomalies** using trained unsupervised deep learning models (Autoencoders) that identify deviations from normal machine behavior.
3. **Escalates confirmed faults** to a multi-agent LangGraph pipeline that reasons about the failure, consults a technical knowledge base, and generates structured repair procedures.
4. **Guides operators interactively** through step-by-step repair workflows in a rich web interface, with full Human-in-the-Loop (HITL) sign-off at every step.
5. **Learns from every resolved incident** by vectorizing the resolution summary and archiving it in an `InteractionMemory` table, which subsequent RAG retrievals draw from — creating a continuously improving knowledge base.

---

## 1.2 Project Directory Structure

```
Gen-AI-Framework-Research-Zynaptrix/
├── Docs/                                 ← Project documentation (human-authored + AI-generated)
│   ├── CODEBASE_ANALYSIS.md
│   ├── GEN_AI_COPILOT_BASELINE.md
│   ├── PROJECT_RESEARCH_REPORT.md
│   └── DETAILED/                         ← This documentation suite
│
└── industrial_copilot/
    ├── backend/                          ← Python FastAPI backend
    │   ├── api/                          ← HTTP + WebSocket route handlers
    │   │   ├── main_api.py               ← FastAPI app entry point
    │   │   ├── anomaly_routes.py         ← Anomaly inference REST routes
    │   │   ├── copilot_chat_api.py       ← Chat history & resolution routes
    │   │   ├── health_routes.py          ← Health check + sensor config
    │   │   └── machine_api.py            ← Machine registry CRUD
    │   │
    │   ├── agents/                       ← Agentic orchestration (LangGraph)
    │   │   ├── copilot_graph.py          ← LangGraph DAG definition
    │   │   ├── knowledge_agent.py        ← RAG-powered knowledge retrieval
    │   │   └── orchestrator_agent.py     ← Top-level agent entrypoint
    │   │
    │   ├── config/                       ← Application configuration
    │   │   ├── settings.py               ← SENSOR_SCHEMA, paths, thresholds
    │   │   └── influx_config.py          ← InfluxDB connection settings
    │   │
    │   ├── database/                     ← Legacy / alternative DB utilities
    │   │   └── neon_vector_store.py
    │   │
    │   ├── ingestion/                    ← Stream listener + InfluxDB writer
    │   │   ├── influx_writer.py          ← InfluxDB write abstraction
    │   │   └── stream_listener.py        ← InfluxDB polling daemon
    │   │
    │   ├── models/                       ← ML model registry + inference
    │   │   ├── detect_anomaly.py         ← AnomalyDetector class
    │   │   └── training/                 ← Autoencoder model training scripts
    │   │
    │   ├── preprocessing/                ← Data preprocessing pipeline
    │   │   ├── data_cleaning.py
    │   │   ├── feature_engineering.py
    │   │   └── normalization.py
    │   │
    │   ├── services/                     ← Application service layer
    │   │   ├── anomaly_service.py        ← Stateful anomaly tracking
    │   │   ├── alert_service.py          ← Alert formatting + logging
    │   │   └── monitoring_service.py     ← End-to-end monitoring loop
    │   │
    │   ├── simulator/                    ← Sensor data generation
    │   │   ├── anomaly_injector.py       ← Fault simulation functions
    │   │   └── sensor_simulator.py       ← Real-time streaming simulator
    │   │
    │   ├── unified_rag/                  ← Multimodal RAG engine
    │   │   ├── config.py                 ← Settings (DB URL, OpenAI key)
    │   │   ├── api/
    │   │   │   └── endpoints.py          ← RAG + machine management endpoints
    │   │   ├── db/
    │   │   │   ├── database.py           ← SQLAlchemy engine + session factory
    │   │   │   └── models.py             ← ORM models (5 tables)
    │   │   ├── embeddings/
    │   │   │   └── embedder.py           ← OpenAI text embedder singleton
    │   │   ├── ingestion/
    │   │   │   ├── captioner.py          ← GPT-4o Vision image captioner
    │   │   │   ├── chunker.py            ← Sliding window text chunker
    │   │   │   ├── parser.py             ← YOLOv8 + PyMuPDF PDF parser
    │   │   │   └── pipeline.py           ← End-to-end ingestion orchestrator
    │   │   └── retrieval/
    │   │       ├── rag.py                ← RAGGenerator (LLM synthesis)
    │   │       └── retriever.py          ← Vector similarity search
    │   │
    │   ├── generate_dataset.py           ← Batch dataset generator
    │   ├── seed_machines.py              ← Machine registry seeder
    │   └── requirements.txt
    │
    └── frontend/                         ← Next.js 14 TypeScript frontend
        └── src/
            ├── app/
            │   ├── layout.tsx            ← Root layout (Redux Provider)
            │   ├── page.tsx              ← Main dashboard (647 lines)
            │   ├── ingestion/            ← Manual upload page
            │   └── machines/             ← Machine registry management
            ├── components/
            │   ├── NavBar.tsx
            │   ├── ProcedureGuide.tsx
            │   └── TaskInteractionCard.tsx
            └── store/
                ├── store.ts              ← Redux store configuration
                ├── Provider.tsx          ← Client-side Redux provider wrapper
                └── slices/
                    ├── copilotSlice.ts   ← Core AI state management (482 lines)
                    ├── machineSlice.ts   ← Machine registry state
                    ├── simulatorSlice.ts ← Simulator control state
                    └── ingestionSlice.ts ← Manual upload state
```

---

## 1.3 Technology Stack

### Backend
| Category | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI | REST endpoints + WebSocket server |
| ASGI Server | Uvicorn | High-performance async server with WebSocket support |
| Data Validation | Pydantic v2 | Request/response model contracts |
| Configuration | pydantic-settings | Env-variable loading with type safety |
| ORM | SQLAlchemy | Database session management and schema |
| Vector Extension | pgvector | Cosine similarity search in PostgreSQL |
| ML Framework | TensorFlow / Keras | Dense and LSTM Autoencoder training and inference |
| Scientific Computing | NumPy, Pandas, scikit-learn | Data manipulation, StandardScaler |
| PDF Parsing | PyMuPDF (fitz) | PDF page-to-pixel rendering and text extraction |
| Layout Detection | YOLOv8 (Ultralytics) | Intelligent document layout segmentation |
| OCR | EasyOCR | Fallback text extraction for image-heavy regions |
| Table Extraction | Camelot | Lattice-based table ripping from PDFs |
| Vision AI | OpenAI GPT-4o Vision | Semantic captioning of technical diagrams |
| Text Embeddings | OpenAI text-embedding-3-small (1536d) | All text → vector conversions |
| LLM Synthesis | OpenAI GPT-4o | Final diagnosis + procedure text generation |
| Agent Orchestration | LangGraph | Stateful multi-node DAG pipeline |
| Agent Utilities | LangChain, LangChain-OpenAI | LLM tool binding and chain creation |
| Time-Series DB | InfluxDB Cloud | 10Hz sensor telemetry storage |
| Vector/Relational DB | PostgreSQL (Neon cloud) + pgvector | Chunks, machines, anomalies, chat |
| Streaming Transport | influxdb-client | InfluxDB read/write SDK |
| HTTP Client | requests | Simulator → API telemetry push |

### Frontend
| Category | Technology | Purpose |
|---|---|---|
| Framework | Next.js 14 (App Router) | File-based routing + SSR/CSR |
| Language | TypeScript 5 | Type-safe state and component contracts |
| State Management | Redux Toolkit v2 | Global application state |
| UI Components | Tailwind CSS v4 | Utility-first styling |
| Icons | Lucide React | Consistent iconography |
| Charts | Recharts | Responsive real-time line charts |
| Markdown Rendering | react-markdown + remark-gfm | Render agent markdown safely |
| Animation | framer-motion | Micro-animations on state transitions |
| API Transport | Native `fetch`, WebSocket | REST calls + WS telemetry stream |

---

## 1.4 Environment Variables Required

The following environment variables must be defined in a `.env` file at the backend root:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# PostgreSQL (Neon)
DATABASE_URL=postgresql://user:password@host/dbname

# InfluxDB Cloud
INFLUX_URL=https://...
INFLUX_TOKEN=...
INFLUX_ORG=...
INFLUX_BUCKET=...

# API URL (used by simulator to push telemetry)
API_URL=http://127.0.0.1:8000
```

On the frontend (Next.js), the backend URL is configured via:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 1.5 System State Machine (Macro View)

The system models machine health as a finite state machine:

```
[NORMAL OPERATION]
       │
       │ MSE > threshold for 3 consecutive ticks
       ▼
[ANOMALY DETECTED]
       │
       │ AnomalyService.consecutive_count >= 3
       ▼
[ESCALATED FAULT]
       │
       │ OrchestratorAgent.handle_anomaly()
       ▼
[AGENTIC INVESTIGATION]
  ┌────┴────┐
  │ LangGraph DAG Pipeline │
  └────┬────┘
       │ Returns final_execution_plan (JSON)
       ▼
[PROCEDURE STREAMING TO UI]
       │
       │ Operator clicks "Done" on each step
       ▼
[INCIDENT RESOLVED]
       │
       │ resolve_incident() endpoint called
       │ GPT-4o summarizes + vectorizes fix
       ▼
[ARCHIVED IN INTERACTION_MEMORY]
       │
       │ Future RAG queries retrieve this fix
       ▼
[NORMAL OPERATION]  ← System is now smarter
```

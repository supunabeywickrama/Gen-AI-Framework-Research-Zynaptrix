# Part 8: Data Flows, End-to-End Traces & Known Issues

## 8.1 Complete Data Flow Diagrams

### 8.1.1 Normal Telemetry Flow (No Anomaly)

```
sensor_simulator.py
  ├─ [1Hz tick]
  ├─ anomaly_injector.normal_reading("PUMP-001")
  │    └─ returns {temperature: 180.1, motor_current: 4.52, ...}
  │
  ├─ [InfluxDB Write]:
  │    InfluxWriter.write_sensor_reading(reading, state="normal")
  │    └─ LINE PROTOCOL: sensor_readings,machine_id=PUMP-001,state=normal
  │         temperature=180.1,motor_current=4.52,...
  │
  └─ [UI Push]:
       requests.post("http://127.0.0.1:8000/api/telemetry/push", json=reading)
         └─ main_api.py: push_telemetry()
              └─ manager.broadcast({"type": "telemetry", "data": reading})
                   └─ [WebSocket to all clients]
                        └─ copilotSlice.ts: ws.onmessage
                             └─ dispatch(addTelemetry({machineId, time, temperature, current, vibration}))
                                  └─ state.telemetry.push(...); if len>20: shift()
                                       └─ Recharts re-renders with new data point
```

### 8.1.2 Anomaly Detection + Escalation Flow

```
stream_listener.py (polling InfluxDB every 1s)
  └─ Flux query: from(bucket) |> range(start:-5s) |> last()
       └─ reads: {temperature: 181, motor_current: 7.2, vibration: 2.8, ...}
            └─ requests.post("http://127.0.0.1:8000/anomaly/detect", json=reading)
                 └─ anomaly_routes.py: detect_anomaly(SensorReading)
                      └─ monitor_service.process_reading(reading_dict, timestamp)
                           └─ anomaly_service.process(reading)
                                └─ AnomalyDetector.detect(reading)
                                     ├─ load_scaler("PUMP-001") → StandardScaler
                                     ├─ X_raw = [181, 7.2, 2.8, 152, 4.1]
                                     ├─ X_norm = scaler.transform(X_raw)
                                     ├─ X_recon = model.predict(X_norm)
                                     ├─ mse = mean((X_norm - X_recon)^2)  ← e.g., 1.42
                                     └─ is_anomaly = (1.42 > 0.7187) = True
                                          └─ AnomalyService:
                                               ├─ consecutive_count = 1, 2, 3... (escalates at 3)
                                               └─ if consecutive_count >= 3:
                                                    ├─ format_alert() → structured alert dict
                                                    ├─ log_alert() → console + anomaly_alerts.jsonl
                                                    └─ _on_anomaly_confirmed(alert)
                                                         └─ OrchestratorAgent.handle_anomaly(data)
                                                              └─ LangGraph workflow.invoke(initial_state)
                                                                   [see §8.1.3]
```

### 8.1.3 LangGraph Pipeline Flow

```
workflow.invoke(initial_state)
  │
  ├─ [Node 1] sensor_status_node
  │    ├─ Read: machine_id, anomaly_score, suspect_sensor, recent_readings
  │    ├─ Classify: score > 1.5*threshold → "FAULT"
  │    ├─ LLM call (GPT-4): "Describe sensor deviations"
  │    └─ Write: sensor_status="FAULT", sensor_analysis="temperature within range, vibration 3.5× normal..."
  │
  ├─ [Node 2] diagnostic_node
  │    ├─ Read: sensor_status, suspect_sensor
  │    ├─ Heuristic: suspect="vibration" + high current → "MECHANICAL"
  │    ├─ LLM call: "Categorize root cause"
  │    └─ Write: diagnostic_category="MECHANICAL", diagnostic_summary="Bearing degradation suspected..."
  │
  ├─ [Node 3] knowledge_node
  │    ├─ Read: machine_id → lookup machines table → manual_id="Zynaptrix_9000"
  │    ├─ Query: "Diagnose MECHANICAL fault. Symptoms: Bearing degradation..."
  │    ├─ embed_text(query) → [1536-dim vector]
  │    ├─ ManualChunk cosine search (top-3 text+table + top-1 image)
  │    ├─ InteractionMemory cosine search for machine_id="PUMP-001" (top-2)
  │    └─ Write: retrieved_knowledge="...bearing replacement procedure...", retrieved_images=["pump_p8_img0.png"]
  │
  ├─ [Node 4] strategy_node
  │    ├─ Read: user_query="Provide quick diagnostic summary"
  │    ├─ Mode detection: query does NOT contain trigger phrase → Mode 1 (Summary)
  │    ├─ Build summary prompt with context
  │    ├─ LLM call (GPT-4o, temp=0.1):
  │    │    → "Based on sensor readings, the elevated vibration (2.8 mm/s) combined with increased
  │    │       motor current (7.2A) strongly indicates bearing wear in the primary drive assembly.
  │    │       [SUGGESTION: Generate full step-by-step repair procedure]"
  │    └─ Write: final_execution_plan = above string
  │
  ├─ [Node 5] critic_node
  │    ├─ Read: final_execution_plan (Mode 1 summary — no LOTO procedure expected)
  │    ├─ Validate: Is it coherent? Does it describe the correct fault? → Yes
  │    └─ Write: critic_approved=True, critic_feedback=""
  │
  └─ route_after_critic → "approved" → END
       └─ result = complete CopilotState dict
            └─ Back in main_api.py: persist ChatMessage rows
                 └─ broadcast anomaly_alert to WebSocket clients
                      └─ frontend dispatches addAnomalyToHistory(...)
```

### 8.1.4 Operator-Triggered Procedure Flow

```
[Frontend: User clicks anomaly in Incident Registry]
  └─ dispatch(setActiveAnomaly(item)); setIsChatOpen(true)
       └─ useEffect triggers: dispatch(fetchChatHistory(anomaly.id))
            ├─ [Empty history] → dispatch(inquireCopilot({query: "Provide quick diagnostic summary..."}))
            │    └─ [sees Mode 1 summary in chat]
            │
            └─ [User clicks "🔧 Start Guided Repair Procedure"]
                 └─ handleManualInquiry("Generate full step-by-step repair procedure")
                      └─ dispatch(inquireCopilot({query: "Generate full step-by-step repair procedure"}))
                           └─ POST /api/copilot/invoke
                                └─ strategy_node detects trigger phrase → Mode 2
                                     └─ GPT-4o generates JSON:
                                          {
                                            "phases": [
                                              { "id": "safety_01", "type": "safety",
                                                "subphases": [{"title": "Pre-Work Safety",
                                                  "tasks": [
                                                    {"id":"s1","text":"Power off and LOTO","critical":true},
                                                    {"id":"s2","text":"Don PPE: gloves, glasses","critical":true}
                                                  ]}]
                                              },
                                              { "id": "maint_01", "type": "maintenance",
                                                "subphases": [{"title": "Bearing Inspection",
                                                  "tasks": [
                                                    {"id":"t1","text":"Remove coupling guard [IMAGE_0]"},
                                                    {"id":"t2","text":"Extract shaft bearing using puller"}
                                                  ]}]
                                              }
                                            ]
                                          }
                                          [wrapped in [PROCEDURE_START]...[PROCEDURE_END]]
                                               └─ copilotSlice: parseProcedureFromContent()
                                                    └─ flattenProcedure() → flatSteps = [s1, s2, t1, t2]
                                                         └─ activeProcedure["42"] = {flatSteps, currentStepIndex:0, ...}
                                                              └─ Pushes phase_header + step(s1) to chatHistory["42"]

[UI renders Step Card for s1: "Power off and LOTO" — AMBER — CRITICAL badge]
  └─ Operator clicks "I have done"
       └─ dispatch(respondToStep({targetId:"42", stepId:"s1", status:"done"}))
            ├─ proc.responses["s1"] = {status:"done"}
            ├─ Find s1 message in history → set stepResponse
            ├─ Push user message: "✅ Completed"
            ├─ proc.currentStepIndex++ → 1
            └─ Push step(s2) to chatHistory["42"]

[Repeats for each step until currentStepIndex >= flatSteps.length]
  └─ proc.completed = true
       └─ Push 'procedure_complete' message: "🎉 All Steps Completed!"

[Operator clicks "Complete Task"]
  └─ setIsResolveModalOpen(true)
       └─ Operator types: "Replaced worn ball bearings at drive end. Used bearing puller #BP-3."
            └─ dispatch(resolveAnomaly({anomalyId:42, operator_fix: "..."}))
                 └─ POST /api/copilot/chat/42/resolve
                      ├─ AnomalyRecord.resolved = True
                      ├─ GPT-4o summarizes chat + fix → "Bearing replacement at drive end housing..."
                      ├─ embed_text(summary) → [1536-dim vector]
                      ├─ InteractionMemory row created with embedding
                      └─ Future PUMP-001 anomalies will retrieve this as a historical fix
```

---

## 8.2 Database Session Management Pattern

The codebase uses two different session patterns:

**Pattern 1: FastAPI Dependency Injection (Preferred)**
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/machines")
async def list_machines(db: Session = Depends(get_db)):
    return db.query(Machine).all()
```

**Pattern 2: Manual Session (Used in agents and services)**
```python
def some_agent_function():
    db = SessionLocal()
    try:
        result = db.query(...).all()
        db.commit()
        return result
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
```

The agents and RAG services cannot use FastAPI's dependency injection system because they run outside the request-response cycle. They must manage sessions manually. This pattern is used correctly throughout the codebase with `finally: db.close()` ensuring no connection leaks.

---

## 8.3 Known Issues & Technical Debt

### 8.3.1 Security — CORS Policy
**Location:** `api/main_api.py`
**Issue:** `allow_origins=["*"]` allows any origin to make credentialed requests.
**Fix:** Replace with `allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:3000")]`

### 8.3.2 Security — No Authentication
**Issue:** All API endpoints are publicly accessible. No JWT, no API key, no session management.
**Fix:** Add FastAPI OAuth2 + JWT middleware. Protect all `/api/copilot/*` and `/ingest-manual` endpoints.

### 8.3.3 Scalability — Global WebSocket Broadcast
**Location:** `api/main_api.py` `TelemetryClientManager`
**Issue:** All connected clients receive telemetry from all machines, regardless of which machine they're viewing.
**Fix:** Implement per-machine subscription rooms: `{machine_id: List[WebSocket]}`. Clients subscribe to specific rooms on connect.

### 8.3.4 Scalability — Synchronous Copilot Invoke
**Location:** `POST /api/copilot/invoke`
**Issue:** The LangGraph workflow runs synchronously on the main async thread, blocking the event loop for 5–30 seconds during a full RAG + LLM pipeline execution.
**Fix:** Run in a thread pool: `await asyncio.get_event_loop().run_in_executor(None, workflow.invoke, state)`

### 8.3.5 Scalability — Global Module Cache
**Location:** `api/anomaly_routes.py` `LATEST_READING_STATE`
**Issue:** Module-level dict is not thread-safe in multi-worker Uvicorn deployments.
**Fix:** Use Redis with a short TTL (`SET latest_reading_pump001 {json} EX 5`).

### 8.3.6 Reliability — No Input Validation on Procedure JSON
**Location:** `copilotSlice.ts` `parseProcedureFromContent()`
**Issue:** If GPT-4o returns malformed JSON inside the `[PROCEDURE_START]` tags, the `JSON.parse()` throws and returns `null`, silently degrading to Mode 1.
**Fix:** Add a JSON Schema validator (e.g., `ajv`) to detect partial procedures and trigger a Critic retry automatically.

### 8.3.7 Testing — Zero Unit Tests
**Issue:** The `requirements.txt` includes `pytest` but no test files exist.
**Priority areas for testing:**
- `AnomalyDetector.detect()` — parametric tests with known MSE inputs/outputs
- `parseProcedureFromContent()` — tests for valid JSON, malformed JSON, missing tags, nested edge cases
- `flattenProcedure()` — ensure correct multi-phase flattening
- `_find_suspect_sensor()` — verify deviation ranking logic

### 8.3.8 Operations — No Schema Migrations
**Issue:** `Base.metadata.create_all()` on startup is not idempotent for schema *changes* — it only creates missing tables, not adds missing columns.
**Fix:** Integrate Alembic. Initialize with `alembic init alembic`, create a migration for each schema change.

### 8.3.9 Observability — No Distributed Tracing
**Issue:** No way to trace a request across the LangGraph nodes or measure per-node latency.
**Fix:** Add OpenTelemetry instrumentation. Tag each LangGraph node with OTLP spans. Export to Jaeger or Honeycomb.

### 8.3.10 Cost Management — Unbounded OpenAI Calls
**Issue:** Every `inquireCopilot` call makes 1 embedding + 1 RAG LLM call. Every ingested image makes 1 vision call. No caching.
**Fix:**
- Cache embeddings for identical query strings (Redis with SHA-256 key)
- Cache RAG results for `(query_hash, manual_id)` pairs with a 1-hour TTL
- Consider using Gemini 1.5 Flash for summary mode and GPT-4o only for procedure mode

---

## 8.4 Operational Runbook

### 8.4.1 Initial Setup

```bash
# 1. Create PostgreSQL database (or use Neon cloud)
# Configure DATABASE_URL in .env

# 2. Install backend dependencies
cd industrial_copilot/backend
pip install -r requirements.txt

# 3. Set up tables (first run)
uvicorn api.main_api:app --reload
# Tables auto-created on startup

# 4. Seed machines
python seed_machines.py

# 5. Generate training dataset (per machine)
python generate_dataset.py --machine_id PUMP-001
python generate_dataset.py --machine_id LATHE-002
python generate_dataset.py --machine_id TURBINE-003

# 6. Train anomaly detection models
python models/training/train_autoencoder.py --machine_id PUMP-001
# Outputs: data/processed/autoencoder_PUMP-001.keras + scaler_PUMP-001.pkl

# 7. Ingest machine manuals (via API)
curl -X POST "http://localhost:8000/ingest-manual" \
  -F "manual_id=Zynaptrix_9000" \
  -F "file=@path/to/pump_manual.pdf"
```

### 8.4.2 Daily Operations

```bash
# Start the backend API
uvicorn api.main_api:app --host 0.0.0.0 --port 8000 --reload

# Start the InfluxDB stream listener (optional)
python -m ingestion.stream_listener

# Start the frontend
cd industrial_copilot/frontend
npm install
npm run dev

# The simulator is controlled via the dashboard UI (Start/Stop buttons)
# Or manually:
python simulator/sensor_simulator.py --machine_id PUMP-001
```

### 8.4.3 Adding a New Machine Type

1. Add base config to `anomaly_injector.get_machine_config()` with machine-specific sensor ranges
2. Generate dataset: `python generate_dataset.py --machine_id MY-MACHINE-001`
3. Train model: `python models/training/train_autoencoder.py --machine_id MY-MACHINE-001`
4. Register machine via API or seed: `POST /machines` with `manual_id` reference
5. Ingest its manual PDF: `POST /ingest-manual`
6. The system is now ready to monitor the new machine

---

## 8.5 Python Backend Dependency Manifest (`requirements.txt`)

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server with WebSocket support |
| `pydantic`, `pydantic-settings` | Data validation and env config |
| `sqlalchemy` | ORM database sessions |
| `psycopg2-binary` | PostgreSQL driver |
| `pgvector` | pgvector SQLAlchemy types |
| `pymupdf` | PDF rendering + text extraction |
| `easyocr` | OCR fallback for dense image regions |
| `camelot-py[cv]` | LaTeX-style table extraction |
| `ultralytics` | YOLOv8 model loading and inference |
| `transformers` | HuggingFace model utilities |
| `torch`, `torchvision` | PyTorch backend for YOLO |
| `opencv-python-headless` | Image processing (Camelot dependency) |
| `huggingface_hub` | Model weight downloads |
| `openai` | GPT-4o, vision, and embedding APIs |
| `langchain`, `langchain-openai`, `langchain-core` | LLM chain utilities |
| `langgraph` | Multi-node agent DAG framework |
| `influxdb-client` | InfluxDB read/write SDK |
| `tensorflow` | Keras Autoencoder training + inference |
| `scikit-learn` | StandardScaler, train/test split |
| `pandas` | DataFrame manipulation |
| `numpy` | Numeric computing |
| `python-multipart` | FastAPI file upload support |
| `python-dotenv` | `.env` file loading |
| `Pillow` | Image file I/O |
| `matplotlib`, `plotly` | Visualization utilities (training/analysis) |
| `pytest` | Unit test framework (not yet implemented) |
| `requests` | Simulator → API HTTP push |
| `sentence-transformers` | Alternative embedding model (legacy) |
| `streamlit` | Legacy monitoring UI (superseded by Next.js) |
| `pypdf` | Auxiliary PDF parsing utilities |

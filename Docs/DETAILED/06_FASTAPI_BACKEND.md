# Part 6: FastAPI Backend — Main App, Routes & WebSocket Layer

## 6.1 Application Bootstrap (`api/main_api.py`)

### 6.1.1 FastAPI Instance Configuration

```python
app = FastAPI(
    title="Zynaptrix Industrial Copilot API",
    version="2.0.0",
)
```

**CORS Middleware:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← Security risk; needs restriction in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
The wildcard CORS policy allows the Next.js dev server (`localhost:3000`) to make cross-origin requests without needing explicit whitelisting. In production this should be replaced with the specific frontend URL.

**Static File Mount:**
```python
app.mount("/static", StaticFiles(directory="data/extracted"), name="static")
```
This mounts the `data/extracted/` directory at `/static/`. All image files extracted from PDFs during ingestion are saved to this directory and served at `http://host:8000/static/{filename}`. The frontend renders `<img src="http://host:8000/static/pump_p3_img0.png">` for inline procedure diagrams.

**Router Registration:**
```python
app.include_router(anomaly_router,  prefix="/anomaly",   tags=["Anomaly Detection"])
app.include_router(health_router,   prefix="/health",    tags=["System Health"])
app.include_router(rag_router,                           tags=["Knowledge Base"])
app.include_router(machine_router,                       tags=["Machine Registry"])
app.include_router(chat_router,                          tags=["Copilot Chat"])
```

Note: `rag_router` and `machine_router` are registered without a prefix because their endpoints already include `/api/` or specific path segments.

### 6.1.2 Startup Event

```python
@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
```

On startup, SQLAlchemy's `create_all()` ensures all tables exist in PostgreSQL. This is a development-friendly pattern — in production, Alembic migrations should handle schema changes instead.

---

## 6.2 WebSocket Telemetry Layer

### 6.2.1 `TelemetryClientManager`

A module-level WebSocket connection manager that broadcasts telemetry to all connected frontend clients:

```python
class TelemetryClientManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)
```

**Current Limitation:** All connected clients receive all telemetry regardless of which machine they're viewing. A scalable implementation would maintain per-machine subscription maps (e.g., `{machine_id: [ws1, ws2]}`).

### 6.2.2 WebSocket Endpoint: `GET /ws/telemetry`

```python
@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive (ping)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
```

The `receive_text()` loop keeps the WebSocket open and allows the client to send heartbeat pings. Data flows exclusively **server → client** (unidirectional broadcast).

### 6.2.3 Telemetry Push Endpoint: `POST /api/telemetry/push`

```python
@app.post("/api/telemetry/push")
async def push_telemetry(reading: dict):
    await manager.broadcast({
        "type": "telemetry",
        "data": reading
    })
    return {"status": "broadcasted"}
```

Called by `sensor_simulator.py` after each tick to propagate the reading to all connected WebSocket clients. The simulator also independently writes to InfluxDB — the push to this endpoint is solely for live UI updates.

**Anomaly Alert Broadcasting:**
When the `OrchestratorAgent` produces a result, it triggers a second broadcast:
```python
await manager.broadcast({
    "type": "anomaly_alert",
    "data": {
        "id":           anomaly_record.id,
        "machine_id":   machine_id,
        "machine_state": state,
        "anomaly_score": score,
        "recent_readings": readings_dict
    }
})
```
The frontend `copilotSlice` handles the `"anomaly_alert"` type to automatically add the incident to the `anomalyHistory` Redux list.

---

## 6.3 Simulator Control Endpoints

The main API provides HTTP endpoints to remotely start/stop machine simulators as subprocess:

### 6.3.1 `POST /api/simulator/start?machine_id=PUMP-001`

```python
process = subprocess.Popen(
    [sys.executable, "simulator/sensor_simulator.py", "--machine_id", machine_id],
    cwd=backend_dir,
)
active_simulators[machine_id] = process
```

Spawns `sensor_simulator.py` as a child process. The PID is stored in `active_simulators: dict[str, subprocess.Popen]`. Multiple machines can be simulated simultaneously.

### 6.3.2 `POST /api/simulator/stop?machine_id=PUMP-001`

```python
process = active_simulators.pop(machine_id)
process.terminate()
process.wait(timeout=5)
```

Sends `SIGTERM` to the simulator process (graceful shutdown). The simulator's `try/except KeyboardInterrupt` block catches the termination and closes the InfluxDB writer connection.

### 6.3.3 `GET /api/simulator/status`

Returns `{"active_simulators": ["PUMP-001", "LATHE-002"]}` — the list of machine IDs with running simulator processes.

---

## 6.4 Machine Registry Endpoints (`api/machine_api.py`)

### 6.4.1 `GET /api/machines`

Queries `db.query(Machine).all()` and returns all registered machines as a list of `MachineResponse` Pydantic models.

### 6.4.2 `GET /api/machines/{machine_id}/anomalies`

Returns all `AnomalyRecord` rows for the specified machine, ordered descending by ID (most recent first). The frontend displays these in the "Incident Registry" panel.

### 6.4.3 `POST /api/machines`

Upsert logic: checks for existing `machine_id` and updates if found, inserts if not.

### 6.4.4 `POST /api/machines/delete/{machine_id}`

Deletes the machine from the registry. Uses `POST` instead of `DELETE` to avoid CORS preflight complexity. Returns `{"status": "decommissioned"}`.

### 6.4.5 `GET /api/chat-history/{anomaly_id}`

Fetches all `ChatMessage` rows linked to `anomaly_id`. Returns a list of message dicts including:
- `role`, `content`, `timestamp`
- `images`: JSON-parsed list of image URLs
- `metadata`: JSON-parsed procedure state (task completion status)
- `db_id`: The database row ID (used for the `PATCH /api/chat-message/{message_id}/task` endpoint)

### 6.4.6 `PATCH /api/chat-message/{message_id}/task`

Updates task completion status within a persistent procedure message:
```python
meta = json.loads(msg.message_metadata) or {}
meta["completed_tasks"][req.task_id] = req.completed
msg.message_metadata = json.dumps(meta)
db.commit()
```
This enables stateful persistence of the operator's step-by-step progress even if the browser refreshes.

### 6.4.7 `POST /api/chat-history/{anomaly_id}/resolve`

The HITL incident resolution endpoint. Complete flow:
1. Sets `AnomalyRecord.resolved = True`
2. Fetches all chat messages and concatenates them into `history_text`
3. Calls GPT-4o with a "Technical Scribe" prompt to summarize the diagnostic exchange and operator fix into ≤150 words
4. Embeds the summary using `text-embedding-3-small`
5. Creates an `InteractionMemory` row with the embedding, summary, and operator fix text
6. Commits the transaction

This is the feedback loop that makes the system learn. Every resolved incident permanently enriches the RAG system's field knowledge.

---

## 6.5 Health Routes (`api/health_routes.py`)

### 6.5.1 `GET /health/status`

Static health check endpoint returning:
```json
{
  "api_status":          "healthy",
  "monitoring_service":  "online",
  "database":            "online",
  "agent_system":        "ready"
}
```

**Note:** This is not a true liveness probe — it doesn't actually validate DB connectivity or LLM availability. A production implementation should query each dependency.

### 6.5.2 `GET /health/sensors`

Returns the `SENSOR_SCHEMA` dict from `config/settings.py`, providing the frontend with normal operating ranges for each sensor. Used to render contextual threshold lines on charts.

### 6.5.3 `GET /health/sensors/{sensor_id}`

Returns config for a specific sensor. Returns 404 if the sensor is not in the schema.

---

## 6.6 Copilot Chat API (`api/copilot_chat_api.py`)

This router provides a secondary path for chat operations (partially overlapping with `machine_api.py` — a historical artifact of iterative development).

### 6.6.1 `GET /api/copilot/chat/{anomaly_id}`

Alternative chat history fetcher. Note that `anomaly_id` comes in as a string; the endpoint tries `int(anomaly_id)` conversion and returns `[]` on failure (handles the `"general"` pseudo-ID case gracefully).

### 6.6.2 `POST /api/copilot/chat/{anomaly_id}/resolve`

Functionally identical to `machine_api.py`'s resolve endpoint. The duplication exists because the frontend's `resolveAnomaly` Redux thunk previously pointed to a different URL prefix. Both endpoints call the same GPT-4o summarization → embedding → `InteractionMemory` archival flow.

---

## 6.7 Complete API Route Map

| Method | Path | Handler | Description |
|---|---|---|---|
| GET | `/ws/telemetry` | WebSocket | Live telemetry stream to UI |
| POST | `/api/telemetry/push` | `push_telemetry` | Simulator pushes readings |
| POST | `/api/copilot/invoke` | `invoke_copilot` | Run LangGraph diagnostic pipeline |
| POST | `/api/simulator/start` | `start_simulator` | Launch machine simulator subprocess |
| POST | `/api/simulator/stop` | `stop_simulator` | Terminate machine simulator |
| GET | `/api/simulator/status` | `get_simulator_status` | List active simulators |
| GET | `/api/machines` | `list_machines` | List registered machines |
| POST | `/api/machines` | `register_machine` | Register or update machine |
| GET | `/api/machines/{id}/anomalies` | `get_machine_anomalies` | Anomaly history for machine |
| POST | `/api/machines/delete/{id}` | `decommission_machine` | Remove machine |
| GET | `/api/chat-history/{id}` | `get_chat_history` | Load conversation for incident |
| PATCH | `/api/chat-message/{id}/task` | `update_task_status` | Mark procedure step done/undone |
| POST | `/api/chat-history/{id}/resolve` | `resolve_incident` | Archive resolved incident |
| POST | `/api/copilot/chat/{id}/resolve` | `resolve_incident` | Alt resolve path |
| POST | `/anomaly/detect` | `detect_anomaly` | Run sensor reading through ML model |
| GET | `/anomaly/history` | `get_history` | Fetch recent anomaly events |
| GET | `/anomaly/latest` | `get_latest_reading` | Cached latest reading state |
| GET | `/health/status` | `system_status` | API health check |
| GET | `/health/sensors` | `get_sensors` | Sensor schema with normal ranges |
| GET | `/health/sensors/{id}` | `get_sensor_info` | Single sensor config |
| POST | `/ingest-manual` | `ingest_manual` | Upload and process PDF manual |
| POST | `/machines` | `create_machine` | RAG-subsystem machine create |
| GET | `/machines` | `list_machines` | RAG-subsystem machine list |
| GET | `/machines/{id}` | `get_machine` | RAG-subsystem single machine |
| POST | `/machines/delete/{id}` | `delete_machine` | RAG-subsystem machine delete |
| POST | `/chat` | `chat` | Direct RAG query |
| GET | `/static/{filename}` | StaticFiles | Serve extracted manual images |

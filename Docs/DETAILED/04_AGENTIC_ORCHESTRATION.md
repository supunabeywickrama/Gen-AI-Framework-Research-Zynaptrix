# Part 4: Agentic Orchestration (LangGraph Pipeline)

## 4.1 Architecture Overview

The agentic orchestration layer is the cognitive core of the Zynaptrix system. When the `MonitoringService` confirms an escalated anomaly, control passes to the `OrchestratorAgent`, which drives a multi-node LangGraph pipeline. The pipeline is deterministic (no LLM-based routing between nodes), meaning the graph always traverses the same sequence unless the Critic node forces a loop.

---

## 4.2 Shared State Object (`CopilotState`)

All nodes communicate exclusively through a shared immutable-accumulation TypedDict called `CopilotState`. No node modifies another node's output — each adds new keys to the state. This is a core LangGraph pattern.

```python
class CopilotState(TypedDict):
    # === INPUT ===
    machine_id:          str          # e.g. "PUMP-001"
    machine_state:       str          # e.g. "machine_fault"
    anomaly_score:       float        # MSE reconstruction error
    suspect_sensor:      str          # e.g. "vibration"
    recent_readings:     dict         # raw sensor values at time of fault
    user_query:          str          # operator's free-text question
    anomaly_id:          int | None   # FK into anomaly_records table

    # === SENSOR STATUS NODE OUTPUT ===
    sensor_status:       str          # "FAULT" | "WARNING" | "NORMAL"
    sensor_analysis:     str          # LLM-generated paragraph

    # === DIAGNOSTIC NODE OUTPUT ===
    diagnostic_category: str          # "MECHANICAL" | "ELECTRICAL" | "SENSOR"
    diagnostic_summary:  str          # structured diagnosis paragraph

    # === KNOWLEDGE NODE OUTPUT ===
    retrieved_knowledge: str          # concatenated RAG text chunks
    retrieved_images:    list[str]    # paths to relevant technical diagrams

    # === STRATEGY NODE OUTPUT ===
    execution_strategy:  str          # proposed maintenance approach
    final_execution_plan: str         # full LLM-generated response (Mode 1 or Mode 2)

    # === CRITIC NODE OUTPUT ===
    critic_approved:     bool         # True = proceed, False = retry strategy
    critic_feedback:     str          # critique text used in strategy retry
```

---

## 4.3 LangGraph DAG Definition (`agents/copilot_graph.py`)

### 4.3.1 Graph Construction

```python
graph = StateGraph(CopilotState)

graph.add_node("sensor_status",    sensor_status_node)
graph.add_node("diagnostic",       diagnostic_node)
graph.add_node("knowledge",        knowledge_node)
graph.add_node("strategy",         strategy_node)
graph.add_node("critic",           critic_node)

graph.set_entry_point("sensor_status")
graph.add_edge("sensor_status", "diagnostic")
graph.add_edge("diagnostic",    "knowledge")
graph.add_edge("knowledge",     "strategy")

# Conditional routing from critic
graph.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "approved": END,
        "retry":    "strategy",
    }
)
graph.add_edge("strategy", "critic")

workflow = graph.compile()
```

### 4.3.2 Conditional Router: `route_after_critic`

```python
def route_after_critic(state: CopilotState) -> str:
    if state.get("critic_approved", False):
        return "approved"
    return "retry"
```

This creates the **retry loop**: if the Critic rejects the strategy (e.g., missing LOTO safety step), the graph routes back to the Strategy node with critic feedback injected into the state for the LLM to address. Maximum 2 retries are enforced in the `OrchestratorAgent.invoke()` wrapper.

---

## 4.4 Node-by-Node Analysis

### 4.4.1 Node 1: `sensor_status_node`

**Purpose:** Ingest the raw alert context and produce an initial severity classification and sensor analysis summary.

**Input fields read:** `machine_id`, `machine_state`, `anomaly_score`, `suspect_sensor`, `recent_readings`

**Logic:**
1. Classifies `sensor_status`:
   - If `anomaly_score > 1.5 * threshold` → `"FAULT"` (critical)
   - If `anomaly_score > threshold` → `"WARNING"`
   - Else → `"NORMAL"` (shouldn't reach here, but defensive)
2. Formats a sensor values table from `recent_readings`
3. Calls the LLM with a targeted system prompt asking for a short sensor status paragraph
4. Writes to state: `sensor_status`, `sensor_analysis`

**Prompt strategy:** Instructs the LLM to describe which sensors are operating outside normal ranges and what physical phenomena might explain the deviation (e.g., "elevated vibration combined with decreased speed suggests bearing friction").

### 4.4.2 Node 2: `diagnostic_node`

**Purpose:** Categorize the root cause of the fault and produce a structured diagnostic summary.

**Input fields read:** `sensor_status`, `sensor_analysis`, `suspect_sensor`, `recent_readings`

**Categories:**
- `"MECHANICAL"` — vibration/speed/current deviations → bearing wear, shaft misalignment, imbalance
- `"ELECTRICAL"` — motor_current anomalies with normal vibration → winding insulation, capacitor, VFD issue
- `"SENSOR"` — freeze or drift patterns detected → transducer calibration drift, wiring fault
- `"THERMAL"` — temperature deviations without strong mechanical signature → cooling system failure, overload

**Logic:**
1. Evaluates the combination of `suspect_sensor` and deviation direction from state
2. Assigns `diagnostic_category` via heuristic rules
3. Calls the LLM to generate a diagnostic paragraph explaining the root cause theory
4. Writes to state: `diagnostic_category`, `diagnostic_summary`

### 4.4.3 Node 3: `knowledge_node`

**Purpose:** Retrieve machine-specific technical knowledge from the unified RAG system.

**Input fields read:** `machine_id`, `diagnostic_category`, `diagnostic_summary`, `user_query`

**Logic:**
1. Looks up `machine_id` in the `machines` table to find its `manual_id`
2. Constructs a contextual query combining the diagnostic summary and user_query:
   ```
   "Diagnose {diagnostic_category} fault. Symptoms: {diagnostic_summary}. Query: {user_query}"
   ```
3. Calls `KnowledgeAgent.query()` which invokes the `RetrievalEngine`, fetching:
   - Top-3 text/table chunks by cosine similarity
   - Top-1 image chunk by cosine similarity  
   - Top-2 historical fixes from `InteractionMemory` matching `machine_id`
4. Assembles the retrieved text into `retrieved_knowledge`
5. Collects image paths into `retrieved_images`
6. Writes to state: `retrieved_knowledge`, `retrieved_images`

**Critical Design: `manual_id` Isolation**
The retriever always filters `ManualChunk.manual_id == manual_id` before performing similarity search. A PUMP-001 anomaly will only search the "Zynaptrix_9000" manual chunks, not LATHE or TURBINE chunks. This prevents completely irrelevant procedures from being served to an operator.

### 4.4.4 Node 4: `strategy_node`

**Purpose:** Synthesize all prior analysis into a final actionable response — either a short diagnostic summary (Mode 1) or a full structured procedure (Mode 2).

**Input fields read:** `diagnostic_summary`, `retrieved_knowledge`, `retrieved_images`, `user_query`, `critic_feedback` (if retry)

**Mode Switching:**
The node checks whether the `user_query` contains the trigger phrase `"Generate full step-by-step repair procedure"` or `"FULL structured JSON repair procedure"`. If yes → Mode 2; else → Mode 1.

**Mode 1 (Summary):**
- Uses `_build_summary_prompt()` from the `RAGGenerator`
- Produces a 3–5 sentence diagnostic paragraph
- Mandates ending with `[SUGGESTION: Generate full step-by-step repair procedure]`
- The suggestion tag is parsed by the frontend to display the "Start Guided Repair Procedure" button

**Mode 2 (Procedure):**
- Uses `_build_procedure_prompt()` from the `RAGGenerator`
- Produces a strict nested JSON structure wrapped in `[PROCEDURE_START]...[PROCEDURE_END]` tags
- Image references `[IMAGE_0]`, `[IMAGE_1]` are embedded inline in task text
- The JSON schema is described exhaustively in the prompt with a literal example

**Critic Feedback Integration:**
If `critic_feedback` exists in state (i.e., this is a retry), the strategy prompt prepends:
```
"PREVIOUS ATTEMPT WAS REJECTED. Critic feedback: {critic_feedback}
Please address this in your new response."
```
This forces the LLM to revise only the criticized aspect while preserving the rest.

**Output written to state:** `final_execution_plan`

### 4.4.5 Node 5: `critic_node`

**Purpose:** Validate the strategy for safety compliance and technical coherence before delivering it to the operator.

**Input fields read:** `final_execution_plan`, `diagnostic_category`

**Validation Rules (checked by LLM in a structured prompt):**
1. For `diagnostic_category == "MECHANICAL"`:
   - Does the procedure include Lockout/Tagout (LOTO) steps?
   - Are PPE requirements listed?
   - Is there a post-repair verification step?
2. For any procedure in Mode 2:
   - Is the first phase type `"safety"`?
   - Do critical tasks have `"critical": true`?
3. General:
   - Is the response coherent and related to the symptoms?
   - Does it avoid contradictions?

**Output written to state:** `critic_approved: bool`, `critic_feedback: str`

---

## 4.5 `OrchestratorAgent` (`agents/orchestrator_agent.py`)

### 4.5.1 `handle_anomaly(data: dict) -> dict`

Top-level entry point called by `MonitoringService`. Constructs an initial `CopilotState` from the anomaly alert data and invokes the compiled LangGraph workflow.

**Execution:**
```python
initial_state = CopilotState(
    machine_id=data["machine_id"],
    machine_state=data["machine_state"],
    anomaly_score=data["anomaly_score"],
    suspect_sensor=data["suspect_sensor"],
    recent_readings=data["recent_readings"],
    user_query="Analyze this machine fault.",  # auto-generated query
    anomaly_id=None,
    ...
)
result = workflow.invoke(initial_state)
```

Returns the fully populated state dict including `final_execution_plan` and all intermediate outputs.

### 4.5.2 `copilot_invoke` API Endpoint (`api/main_api.py`)

**`POST /api/copilot/invoke`** is the REST endpoint called by the frontend `inquireCopilot` Redux thunk.

**Request body:**
```json
{
  "machine_id":      "PUMP-001",
  "user_query":      "Generate full step-by-step repair procedure",
  "machine_state":   "machine_fault",
  "anomaly_id":      42,
  "recent_readings": { "temperature": 181, "motor_current": 7.2, ... },
  "suspect_sensor":  "Operator-Triggered Context"
}
```

**Processing:**
1. Builds `CopilotState` from request
2. Calls `workflow.invoke()` synchronously (important: NOT in an async executor — this is a known blocking design choice)
3. Persists agent messages to `ChatMessage` table in PostgreSQL
4. Returns `graph_result` dict containing `final_execution_plan` and `retrieved_images`

**Database Persistence:**
On each copilot invocation, both the user message and agent response are saved as `ChatMessage` rows linked to the `anomaly_id`. This enables chat history restoration on page refresh.

---

## 4.6 `KnowledgeAgent` (`agents/knowledge_agent.py`)

A standalone agent that wraps the unified RAG system and exposes a single `query()` method.

### 4.6.1 Constructor

```python
self.rag_generator = RAGGenerator()
```

Instantiates the `RAGGenerator` from `unified_rag/retrieval/rag.py`. This is a lazy-loaded singleton to avoid repeated connection setup.

### 4.6.2 `query(machine_id, question, manual_id, diagnostic_context) -> dict`

1. Resolves `manual_id` from the `machines` table if not provided
2. Calls `self.rag_generator.generate_response(question, manual_id, machine_id)`
3. Returns the structured response: `{ "answer": str, "images": list[str], "pages": list[int] }`

### 4.6.3 Path Normalization

The agent contains a critical cross-platform path normalization step. Image paths stored in PostgreSQL may use Windows backslashes (`\`) but need to be served as forward-slash URL paths for the `/static/` FastAPI mount:

```python
image_paths = [
    p.replace("\\", "/").replace("data/extracted/", "")
    for p in raw_paths
]
```

This ensures `data\extracted\manual_p3_img0.png` becomes `manual_p3_img0.png`, which is then served at `http://localhost:8000/static/manual_p3_img0.png`.

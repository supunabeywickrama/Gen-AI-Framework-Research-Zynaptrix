# Part 7: Frontend — Next.js Dashboard, Redux State, & UI Interactions

## 7.1 Architecture Overview

The frontend is a **Next.js 14** application using the App Router paradigm. It is structured as a client-side-heavy application where the bulk of logic lives in Redux Toolkit slices. The server-side rendering capability of Next.js is largely unused — most components are marked `"use client"`.

**State Management Philosophy:** All server state (machine list, anomaly history, chat messages) is managed in Redux and fetched via `createAsyncThunk` actions. All transient UI state (modal open/closed, input field values, step comments) is kept in component-local `useState`.

---

## 7.2 Redux Store Architecture (`store/`)

### 7.2.1 `store.ts` — Store Configuration

```typescript
export const store = configureStore({
  reducer: {
    copilot:   copilotReducer,    // AI state, telemetry, chat, anomaly
    machines:  machineReducer,    // Fleet registry
    simulator: simulatorReducer,  // Active simulators
    ingestion: ingestionReducer,  // PDF upload state
  },
});
```

Four top-level slices. The `copilot` slice is the most complex (482 lines) and acts as the single source of truth for the entire diagnostic workflow.

---

## 7.3 `copilotSlice.ts` — Core AI State (482 lines)

### 7.3.1 Type Definitions

**`TelemetryPoint`:** Represents a single WebSocket-received sensor reading for one machine.
```typescript
interface TelemetryPoint {
  machineId: string;
  time: string;         // formatted time string for chart X axis
  temperature: number;
  current: number;      // mapped from motor_current
  vibration: number;
}
```

**`ProcedureTask`:** A single atomic step within a repair procedure.
```typescript
interface ProcedureTask {
  id: string;            // e.g., "s1", "t3"
  text: string;          // step instruction (may contain [IMAGE_N] refs)
  critical?: boolean;    // true = safety-critical, shows red warning badge
}
```

**`ProcedureSubPhase`** → groups tasks with a shared sub-title (e.g., "Pre-Work Safety")
**`ProcedurePhase`** → groups sub-phases under a shared title and type ("safety" or "maintenance")
**`ProcedureData`** → top-level container: `{ phases: ProcedurePhase[] }`

**`StepData`:** Runtime metadata attached to each step message in the chat UI.
```typescript
interface StepData {
  stepId:        string;   // matches ProcedureTask.id
  phaseType:     string;   // "safety" | "maintenance" — controls card color
  phaseTitle:    string;   // phase section name
  subphaseTitle: string;   // subphase section name
  stepIndex:     number;   // 0-based position in flat step array
  totalSteps:    number;   // total steps count for progress display
  critical?:     boolean;
}
```

**`StepResponse`:** The operator's response to a step.
```typescript
interface StepResponse {
  status: 'done' | 'undone' | 'cant_do' | 'havent_done';
  comment?: string;
}
```

**`FlatStep`:** Internal representation after flattening the multi-level procedure tree into a linear array. Each `FlatStep` includes its phase and subphase context so the UI can render section headers on transitions.

**`ActiveProcedure`:** The in-memory state machine tracking progress through a procedure for a given anomaly session.
```typescript
interface ActiveProcedure {
  flatSteps:        FlatStep[];              // ordered flat list of all steps
  currentStepIndex: number;                 // which step is active
  responses:        Record<string, StepResponse>; // stepId → response
  images:           string[];               // image URLs from RAG
  completed:        boolean;
}
```

**`ChatMessage`:** Extended message type supporting multiple rendering modes.
```typescript
interface ChatMessage {
  role:          'agent' | 'user';
  content:       string;
  images?:       string[];
  machineId?:    string;
  quickActions?: string[];
  dbId?:         number;          // PostgreSQL row ID for mutation endpoints
  type?:         'text' | 'step' | 'phase_header' | 'procedure_complete';
  stepData?:     StepData;        // set when type === 'step'
  stepResponse?: StepResponse;    // set after operator responds
}
```

**`CopilotState`:** Top-level Redux state shape.
```typescript
interface CopilotState {
  telemetry:      TelemetryPoint[];              // Rolling 20-point buffer
  chatHistory:    Record<string, ChatMessage[]>; // keyed by anomaly ID or "general"
  loadingHistory: Record<string, boolean>;       // loading state per anomaly
  anomalyHistory: AnomalyRecord[];               // all known anomalies for current machine
  activeAnomaly:  AnomalyRecord | null;          // currently selected incident
  systemState:    'NORMAL' | 'ANOMALY';
  anomalyScore:   number;
  activeAgents:   string[];
  fleetHealth:    Record<string, number>;
  activeProcedure: Record<string, ActiveProcedure>; // keyed by anomaly ID
}
```

### 7.3.2 Helper Functions

**`parseProcedureFromContent(content: string) -> ProcedureData | null`**
Extracts JSON from between `[PROCEDURE_START]` and `[PROCEDURE_END]` tags and parses it. Returns `null` if tags not found or JSON.parse fails. This determines which rendering mode the UI uses.

**`flattenProcedure(procedure: ProcedureData) -> FlatStep[]`**
Three-level loop: phases → subphases → tasks. Each task becomes a `FlatStep` enriched with its phase and subphase context. This linear array is what `respondToStep` uses to advance the workflow.

### 7.3.3 Async Thunks

**`fetchAnomalyHistory(machineId)`**
`GET /api/machines/{machineId}/anomalies` → populates `anomalyHistory`

**`fetchChatHistory(anomalyId)`**
`GET /api/chat-history/{anomalyId}` → populates `chatHistory[anomalyId.toString()]`

Returns `{anomalyId, messages}` so the fulfilled reducer can key the messages correctly.

**`resolveAnomaly({anomalyId, operator_fix})`**
`POST /api/copilot/chat/{anomalyId}/resolve` with body `{operator_fix}`. On success, marks the anomaly as resolved in local state and appends an archive confirmation message.

**`inquireCopilot({machine_id, query, machine_state, context_anomaly})`**

This is the most critical thunk. Full execution flow:
1. Dispatches `addChatMessage` with the user message immediately (optimistic UI)
2. Dispatches `addChatMessage` with `"⏳ Analyzing incident context..."` placeholder
3. POSTs to `/api/copilot/invoke` with the full context
4. On fulfillment, the `extraReducers` handler:
   - Calls `parseProcedureFromContent(result.final_execution_plan)`
   - **If procedure found (Mode 2):**
     - Removes the placeholder message with `history.splice(lastIdx, 1)`
     - Calls `flattenProcedure(procedure)` and stores in `activeProcedure[targetId]`
     - Pushes a `phase_header` message for the first phase
     - Pushes the first `step` message with full `stepData`
   - **If no procedure (Mode 1):**
     - Replaces placeholder with the summary text
     - Preserves `[SUGGESTION: ...]` tag if present in content

### 7.3.4 `respondToStep` Reducer

This is a synchronous reducer (no API call) that handles the operator clicking "Done", "Undone", etc.:

```typescript
respondToStep(state, action: PayloadAction<{
  targetId: string;
  stepId: string;
  status: string;
  comment?: string;
}>) {
  const proc = state.activeProcedure[targetId];
  // 1. Record response in proc.responses
  // 2. Find the step message in chatHistory and set stepResponse on it
  // 3. Push user confirmation message
  // 4. Increment currentStepIndex
  // 5a. If all steps done → push 'procedure_complete' message
  // 5b. If phase transition → push 'phase_header' message, then next 'step' message
}
```

**Phase transition detection:** Compares `nextStep.phaseId !== prevStep.phaseId`. When true, injects a phase completion header before the next step card. This creates the visual "chapter breaks" in the chat UI.

---

## 7.4 `machineSlice.ts`

**State:** `{machines: Machine[], currentMachineId: string, loading: bool, error: string|null}`

**Thunks:**
- `fetchMachines()` → `GET /api/machines`
- `registerMachine(machine)` → `POST /api/machines`
- `deleteMachine(machineId)` → `POST /api/machines/delete/{id}`

`setCurrentMachineId(id)` is a synchronous action that switches the active machine context. When dispatched, `page.tsx` uses a `useEffect` to reload the anomaly history for the new machine.

**Upsert logic in fulfilled handler:**
```typescript
const index = state.machines.findIndex(m => m.machine_id === action.payload.machine_id);
if (index !== -1) state.machines[index] = action.payload;
else state.machines.push(action.payload);
```

### 7.4.1 `simulatorSlice.ts`

**State:** `{activeSimulators: string[], loading: bool, error: string|null}`

- `fetchSimulatorStatus()` → `GET /api/simulator/status` → returns `string[]` of active machine IDs
- `startSimulator(machineId)` → `POST /api/simulator/start?machine_id=...`
- `stopSimulator(machineId)` → `POST /api/simulator/stop?machine_id=...`

### 7.4.2 `ingestionSlice.ts`

**State:** `{isUploading: bool, uploadStatus: string|null, error: string|null}`

`uploadManual({manualId, file})` sends `multipart/form-data` to `/ingest-manual`. On success, dispatches `addChatMessage` to the copilot slice to notify the operator.

---

## 7.5 Main Dashboard (`app/page.tsx` — 647 lines)

### 7.5.1 Component State

**Redux state (via `useSelector`):**
- `telemetry`, `chatHistory`, `anomalyHistory`, `activeAnomaly`, `systemState`, `anomalyScore`, `loadingHistory`, `activeProcedure` from `state.copilot`
- `machines`, `currentMachineId` from `state.machines`
- `activeSimulators` from `state.simulator`

**Local state (via `useState`):**
- `query` — text input for manual copilot queries
- `isChatOpen` — controls chat modal visibility
- `isChatMaximized` — toggles chat modal between floating and fullscreen
- `isResolveModalOpen` — controls incident resolution dialog
- `operatorFix` — textarea value for the operator's resolution notes
- `stepComments` — `Record<string, string>` for per-step comment inputs
- `isMounted` — prevents Recharts SSR hydration mismatches

### 7.5.2 Effects

**Mount effect:**
```typescript
useEffect(() => {
  setIsMounted(true);
  dispatch(fetchMachines());
  dispatch(fetchSimulatorStatus());
}, [dispatch]);
```

**Machine change effect:**
```typescript
useEffect(() => {
  if (currentMachineId) {
    dispatch(fetchAnomalyHistory(currentMachineId));
  }
}, [currentMachineId, dispatch]);
```

**Anomaly selection effect:**
```typescript
useEffect(() => {
  if (activeAnomaly) {
    dispatch(fetchChatHistory(activeAnomaly.id)).then((action) => {
      if (action.payload?.messages.length === 0) {
        // Auto-trigger initial diagnosis
        dispatch(inquireCopilot({
          machine_id: activeAnomaly.machine_id,
          query: "Provide a quick diagnostic summary for this alert.",
          machine_state: activeAnomaly.type,
          context_anomaly: activeAnomaly
        }));
      }
    });
  }
}, [activeAnomaly?.id, dispatch]);
```

**WebSocket effect:**
```typescript
useEffect(() => {
  const connectWS = () => {
    const wsUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
      .replace('http', 'ws');
    const ws = new WebSocket(`${wsUrl}/ws/telemetry`);

    ws.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      if (parsed.type === "telemetry") {
        dispatch(addTelemetry({ machineId, time, temperature, current, vibration }));
      } else if (parsed.type === "anomaly_alert") {
        dispatch(setSystemState('ANOMALY'));
        dispatch(setAnomalyScore(...));
        dispatch(addAnomalyToHistory({...}));
      }
    };

    ws.onclose = () => setTimeout(connectWS, 3000); // Auto-reconnect after 3s
  };
  connectWS();
  return () => ws?.close();   // Cleanup on unmount
}, [dispatch]);
```

The `ws.onclose` auto-reconnect logic ensures the dashboard recovers from temporary backend disconnections without manual page refresh.

### 7.5.3 Telemetry Chart

```typescript
const filteredTelemetry = telemetry.filter(t => t.machineId === currentMachineId);
```

The telemetry array stores readings for ALL machines but the chart renders only the current machine's data. The rolling buffer is capped at 20 points (`state.telemetry.shift()` when length > 20).

Recharts `<LineChart>` configuration:
- 3 data series: `temperature` (blue), `current` (emerald), `vibration` (amber)
- `animationDuration={300}` for smooth real-time updates
- No dots (`dot={false}`) for performance with high-frequency data
- Custom `<Tooltip>` with dark slate background matching the dashboard theme

### 7.5.4 Message Rendering Strategy

The chat modal renders messages with a **type-dispatched rendering pattern**:

| Message `type` | Component Rendered | Trigger |
|---|---|---|
| `"phase_header"` | Centered gradient banner | Phase transitions in procedure |
| `"step"` | Interactive step card (blue or amber) | Each procedural step |
| `"procedure_complete"` | Celebration banner with finalize prompt | All steps completed |
| `role === "user"` | Right-aligned blue bubble | Manual query or step response |
| (default) | Left-aligned agent bubble | Summary text, RAG answers |

**`renderStepContent(text, images)`**
Parses `[IMAGE_0]`, `[IMAGE_1]` etc. from task text using regex `/\[IMAGE[_\s-]?\d+\]/gi` and replaces them with `<img>` elements sourced from the `images` array. Non-image segments are rendered as plain `<span>`.

**Safety vs Maintenance step styling:**
Safety phase steps (`phaseType === "safety"`) render with amber color scheme and three-option buttons (Done / Haven't Done / Can't Do). Maintenance phase steps use blue scheme and an optional comment textarea plus two-option buttons (Done / Undone).

### 7.5.5 Suggestion Button Mechanism

The agent summary response (Mode 1) ends with the LLM-generated tag:
```
[SUGGESTION: Generate full step-by-step repair procedure]
```

The rendering code detects `msg.content.includes('[SUGGESTION:')` and renders a "Start Guided Repair Procedure" button:
```typescript
<button
  onClick={() => handleManualInquiry("Generate full step-by-step repair procedure")}
>
  🔧 Start Guided Repair Procedure
</button>
```

Clicking this button dispatches `inquireCopilot` with the exact trigger phrase, causing the backend to switch to Mode 2 and return the structured JSON procedure.

### 7.5.6 Resolution Modal

When "Complete Task" is clicked:
1. `isResolveModalOpen = true`
2. A modal renders a `<textarea>` for `operatorFix`
3. "Archive Incident" button dispatches `resolveAnomaly({anomalyId, operator_fix})`
4. Redux thunk POSTs to resolve endpoint; on success, marks `resolved = true` in both `anomalyHistory` and `activeAnomaly`
5. The chat input becomes disabled: `readOnly={activeAnomaly?.resolved}`
6. A confirmation message is appended to the chat

---

## 7.6 Frontend Dependencies (`package.json`)

| Package | Version | Purpose |
|---|---|---|
| `next` | 16.2.1 | App Router framework |
| `react` | 19.2.4 | Component runtime |
| `@reduxjs/toolkit` | ^2.11.2 | State management |
| `react-redux` | ^9.2.0 | React-Redux bindings |
| `recharts` | ^3.8.0 | Chart library |
| `react-markdown` | ^10.1.0 | Markdown → JSX |
| `remark-gfm` | ^4.0.1 | GitHub Flavored Markdown plugin |
| `lucide-react` | ^0.577.0 | Icon library |
| `framer-motion` | ^12.38.0 | Animation primitives |
| `tailwindcss` | ^4 | CSS utility framework |
| `clsx` | ^2.1.1 | Conditional class name utility |
| `tailwind-merge` | ^3.5.0 | Merge Tailwind classes safely |
| `typescript` | ^5 | Type checking |

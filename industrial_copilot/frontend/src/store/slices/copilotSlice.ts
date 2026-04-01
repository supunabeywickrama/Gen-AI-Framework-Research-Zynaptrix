import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface TelemetryPoint {
  machineId: string;
  time: string;
  temperature: number;
  current: number;
  vibration: number;
}

// Structured Procedure Types
export interface ProcedureTask {
  id: string;
  text: string;
  critical?: boolean;
}

export interface ProcedureSubPhase {
  title: string;
  tasks: ProcedureTask[];
}

export interface ProcedurePhase {
  id: string;
  title: string;
  type?: string; // 'safety' | 'maintenance'
  subphases: ProcedureSubPhase[];
}

export interface ProcedureData {
  phases: ProcedurePhase[];
}

// === STEP-BY-STEP INTERACTIVE TYPES ===
export interface StepData {
  stepId: string;
  phaseType: string;
  phaseTitle: string;
  subphaseTitle: string;
  stepIndex: number;
  totalSteps: number;
  critical?: boolean;
}

export interface StepResponse {
  status: 'done' | 'undone' | 'cant_do' | 'havent_done';
  comment?: string;
}

export interface FlatStep {
  id: string;
  text: string;
  phaseId: string;
  phaseTitle: string;
  phaseType: string;
  subphaseTitle: string;
  critical?: boolean;
}

export interface ActiveProcedure {
  flatSteps: FlatStep[];
  currentStepIndex: number;
  responses: Record<string, StepResponse>;
  images: string[];
  completed: boolean;
}

export interface ChatMessage {
  role: 'agent' | 'user';
  content: string;
  images?: string[];
  machineId?: string;
  quickActions?: string[];
  dbId?: number;
  // Step-by-step interactive
  type?: 'text' | 'step' | 'phase_header' | 'procedure_complete';
  stepData?: StepData;
  stepResponse?: StepResponse;
}

export interface AnomalyRecord {
  id: number;
  machine_id: string;
  timestamp: string;
  type: string;
  score: number;
  sensor_data: string;
  resolved?: boolean;
}

interface CopilotState {
  telemetry: TelemetryPoint[];
  chatHistory: Record<string, ChatMessage[]>;
  loadingHistory: Record<string, boolean>;
  anomalyHistory: AnomalyRecord[];
  activeAnomaly: AnomalyRecord | null;
  systemState: 'NORMAL' | 'ANOMALY';
  anomalyScore: number;
  activeAgents: string[];
  fleetHealth: Record<string, number>;
  activeProcedure: Record<string, ActiveProcedure>;
}

const initialState: CopilotState = {
  telemetry: [],
  chatHistory: {
    'general': [
      { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring multi-machine factory floor.' }
    ]
  },
  loadingHistory: {},
  anomalyHistory: [],
  activeAnomaly: null,
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
  fleetHealth: {},
  activeProcedure: {},
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// --- Parse procedure JSON from agent response ---
function parseProcedureFromContent(content: string): ProcedureData | null {
  const startTag = '[PROCEDURE_START]';
  const endTag = '[PROCEDURE_END]';
  const startIdx = content.indexOf(startTag);
  const endIdx = content.indexOf(endTag);
  if (startIdx === -1 || endIdx === -1) return null;
  try {
    const jsonStr = content.substring(startIdx + startTag.length, endIdx).trim();
    return JSON.parse(jsonStr) as ProcedureData;
  } catch {
    return null;
  }
}

// --- Flatten a procedure into linear steps ---
function flattenProcedure(procedure: ProcedureData): FlatStep[] {
  const steps: FlatStep[] = [];
  procedure.phases.forEach(phase => {
    phase.subphases.forEach(sp => {
      sp.tasks.forEach(t => {
        steps.push({
          id: t.id,
          text: t.text,
          phaseId: phase.id,
          phaseTitle: phase.title,
          phaseType: phase.type || 'maintenance',
          subphaseTitle: sp.title,
          critical: t.critical,
        });
      });
    });
  });
  return steps;
}

export const fetchAnomalyHistory = createAsyncThunk(
  'copilot/fetchAnomalies',
  async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/machines/${machineId}/anomalies`);
    if (!response.ok) throw new Error('Failed to fetch anomaly history');
    return await response.json();
  }
);

export const fetchChatHistory = createAsyncThunk(
  'copilot/fetchChat',
  async (anomalyId: number) => {
    const response = await fetch(`${API_BASE}/api/chat-history/${anomalyId}`);
    if (!response.ok) throw new Error('Failed to fetch chat history');
    return { anomalyId, messages: await response.json() };
  }
);

export const resolveAnomaly = createAsyncThunk(
  'copilot/resolve',
  async (payload: { anomalyId: number; operator_fix: string }) => {
    const response = await fetch(`${API_BASE}/api/copilot/chat/${payload.anomalyId}/resolve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operator_fix: payload.operator_fix }),
    });
    if (!response.ok) throw new Error('Failed to resolve incident');
    return { anomalyId: payload.anomalyId, data: await response.json() };
  }
);

export const inquireCopilot = createAsyncThunk(
  'copilot/inquire',
  async (payload: { machine_id: string; query: string; machine_state: string; context_anomaly?: AnomalyRecord }, { dispatch }) => {
    const targetId = payload.context_anomaly?.id.toString();
    
    // Add user message first
    dispatch(addChatMessage({ 
        role: 'user', 
        content: payload.query, 
        machineId: payload.machine_id,
        targetId 
    }));
    
    // Add a placeholder agent message
    dispatch(addChatMessage({ 
        role: 'agent', 
        content: '⏳ Analyzing incident context...',
        targetId
    }));

    const body = {
        machine_id: payload.machine_id,
        user_query: payload.query,
        machine_state: payload.machine_state,
        anomaly_id: payload.context_anomaly?.id,
        recent_readings: payload.context_anomaly ? JSON.parse(payload.context_anomaly.sensor_data) : {},
        suspect_sensor: "Operator-Triggered Context"
    };

    const response = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!response.ok) throw new Error('Copilot inquiry failed');
    const data = await response.json();
    return data;
  }
);

const copilotSlice = createSlice({
  name: 'copilot',
  initialState,
  reducers: {
    addTelemetry(state, action: PayloadAction<TelemetryPoint>) {
      state.telemetry.push(action.payload);
      if (state.telemetry.length > 20) state.telemetry.shift();
    },
    addChatMessage(state, action: PayloadAction<ChatMessage & { targetId?: string }>) {
      const { targetId, ...msg } = action.payload;
      const key = targetId || 'general';
      if (!state.chatHistory[key]) {
        state.chatHistory[key] = [];
      }
      state.chatHistory[key].push(msg);
    },
    addAnomalyToHistory(state, action: PayloadAction<AnomalyRecord>) {
      state.anomalyHistory.unshift(action.payload);
      if (state.anomalyHistory.length > 50) state.anomalyHistory.pop();
    },
    setActiveAnomaly(state, action: PayloadAction<AnomalyRecord | null>) {
      state.activeAnomaly = action.payload;
    },
    setSystemState(state, action: PayloadAction<'NORMAL' | 'ANOMALY'>) {
      state.systemState = action.payload;
    },
    setAnomalyScore(state, action: PayloadAction<number>) {
      state.anomalyScore = action.payload;
    },
    setActiveAgents(state, action: PayloadAction<string[]>) {
      state.activeAgents = action.payload;
    },

    // === NEW: Step-by-step procedure interaction ===
    respondToStep(state, action: PayloadAction<{ targetId: string; stepId: string; status: string; comment?: string }>) {
      const { targetId, stepId, status, comment } = action.payload;
      const proc = state.activeProcedure[targetId];
      if (!proc) return;

      const history = state.chatHistory[targetId];
      if (!history) return;

      // Record response in procedure state
      proc.responses[stepId] = { status: status as StepResponse['status'], comment };

      // Mark the step message as responded (find the last step message with this ID)
      for (let i = history.length - 1; i >= 0; i--) {
        if (history[i].type === 'step' && history[i].stepData?.stepId === stepId) {
          history[i].stepResponse = { status: status as StepResponse['status'], comment };
          break;
        }
      }

      // Add user response message to chat
      const labels: Record<string, string> = {
        'done': '✅ Completed',
        'undone': '❌ Not completed',
        'cant_do': "⚠️ Unable to perform",
        'havent_done': "⏳ Not done yet",
      };
      let userMsg = labels[status] || status;
      if (comment && comment.trim()) userMsg += `\n💬 "${comment}"`;
      history.push({ role: 'user', content: userMsg });

      // Advance to next step
      proc.currentStepIndex++;

      if (proc.currentStepIndex >= proc.flatSteps.length) {
        // === ALL STEPS DONE ===
        proc.completed = true;
        history.push({
          role: 'agent',
          content: '🎉 **All procedure steps completed!**\n\nGreat work! Click **Complete Task** above to finalize and archive this incident with your notes.',
          type: 'procedure_complete',
        });
      } else {
        const nextStep = proc.flatSteps[proc.currentStepIndex];
        const prevStep = proc.flatSteps[proc.currentStepIndex - 1];

        // Phase transition header
        if (nextStep.phaseId !== prevStep.phaseId) {
          history.push({
            role: 'agent',
            content: `✅ **${prevStep.phaseTitle}** — Complete!\n\nProceeding to **${nextStep.phaseTitle}**...`,
            type: 'phase_header',
          });
        }

        // Add next step message
        history.push({
          role: 'agent',
          content: nextStep.text,
          type: 'step',
          stepData: {
            stepId: nextStep.id,
            phaseType: nextStep.phaseType,
            phaseTitle: nextStep.phaseTitle,
            subphaseTitle: nextStep.subphaseTitle,
            stepIndex: proc.currentStepIndex,
            totalSteps: proc.flatSteps.length,
            critical: nextStep.critical,
          },
          images: proc.images,
        });
      }
    },
  },
  extraReducers: (builder) => {
    builder.addCase(fetchAnomalyHistory.fulfilled, (state, action) => {
      state.anomalyHistory = action.payload;
    });
    
    // Chat Persistence Loading
    builder.addCase(fetchChatHistory.pending, (state, action) => {
        state.loadingHistory[action.meta.arg.toString()] = true;
    });
    builder.addCase(fetchChatHistory.fulfilled, (state, action) => {
        const { anomalyId, messages } = action.payload;
        if (messages.length === 0) {
            // Empty — the auto-diagnosis useEffect in page.tsx will populate
            state.chatHistory[anomalyId.toString()] = [];
        } else {
            // Hydrate messages from DB (text messages only for persistence)
            state.chatHistory[anomalyId.toString()] = messages.map((m: any) => {
                return {
                    role: m.role,
                    content: m.content,
                    images: m.images || [],
                    dbId: m.db_id
                };
            });
        }
        state.loadingHistory[anomalyId.toString()] = false;
    });
    builder.addCase(fetchChatHistory.rejected, (state, action) => {
        const anomalyId = action.meta.arg.toString();
        state.loadingHistory[anomalyId] = false;
        if (!state.chatHistory[anomalyId]) {
            state.chatHistory[anomalyId] = [
                { role: 'agent', content: '⚠️ Could not restore past history. Diagnostic bridge active.' }
            ];
        }
    });
    
    // Resolution Logic
    builder.addCase(resolveAnomaly.fulfilled, (state, action) => {
        const { anomalyId, data } = action.payload;
        const record = state.anomalyHistory.find(r => r.id === anomalyId);
        if (record) record.resolved = true;
        if (state.activeAnomaly?.id === anomalyId) state.activeAnomaly.resolved = true;
        
        state.chatHistory[anomalyId.toString()].push({
            role: 'agent',
            content: `✅ **Incident Resolved & Archived**\n\n**Action Summary:**\n${data.summary}`
        });
    });

    // === MAIN HANDLER: Copilot Response ===
    builder.addCase(inquireCopilot.fulfilled, (state, action) => {
      const targetId = action.meta.arg.context_anomaly?.id.toString() || 'general';
      const history = state.chatHistory[targetId];
      if (!history || history.length === 0) return;

      const lastIdx = history.length - 1;
      const result = action.payload.graph_result;
      const content = result.final_execution_plan;
      const procedure = parseProcedureFromContent(content);

      if (procedure) {
        // === PROCEDURE RECEIVED: Start step-by-step flow ===
        // Remove placeholder "Analyzing..." message
        history.splice(lastIdx, 1);

        // Flatten steps
        const flatSteps = flattenProcedure(procedure);

        // Store active procedure
        state.activeProcedure[targetId] = {
          flatSteps,
          currentStepIndex: 0,
          responses: {},
          images: result.retrieved_images || [],
          completed: false,
        };

        // Add phase header + first step
        if (flatSteps.length > 0) {
          const firstStep = flatSteps[0];
          history.push({
            role: 'agent',
            content: `**${firstStep.phaseTitle}**\n\nLet's begin with the safety protocols. I'll guide you through each step one by one.`,
            type: 'phase_header',
          });
          history.push({
            role: 'agent',
            content: firstStep.text,
            type: 'step',
            stepData: {
              stepId: firstStep.id,
              phaseType: firstStep.phaseType,
              phaseTitle: firstStep.phaseTitle,
              subphaseTitle: firstStep.subphaseTitle,
              stepIndex: 0,
              totalSteps: flatSteps.length,
              critical: firstStep.critical,
            },
            images: result.retrieved_images,
          });
        }
      } else {
        // === SUMMARY/TEXT MODE (no procedure) ===
        // Clean content: strip tags for display
        const cleanContent = content
          .replace(/\[SUGGESTION:.*?\]/gi, '')
          .trim();

        const hasSuggestion = content.includes('[SUGGESTION:');

        history[lastIdx] = {
          role: 'agent',
          content: hasSuggestion ? cleanContent + '\n\n[SUGGESTION: Generate full step-by-step repair procedure]' : cleanContent,
          images: result.retrieved_images,
          machineId: action.meta.arg.machine_id,
          dbId: action.payload.db_id,
        };
      }
    });

    builder.addCase(inquireCopilot.rejected, (state, action) => {
       const targetId = action.meta.arg.context_anomaly?.id.toString() || 'general';
       const history = state.chatHistory[targetId];
       if (history && history.length > 0) {
         const lastIdx = history.length - 1;
         history[lastIdx].content = "⚠️ Error communicating with Copilot backend.";
       }
    });
  }
});

export const { 
  addTelemetry, 
  addChatMessage, 
  addAnomalyToHistory,
  setActiveAnomaly,
  setSystemState, 
  setAnomalyScore,
  setActiveAgents,
  respondToStep
} = copilotSlice.actions;

export default copilotSlice.reducer;

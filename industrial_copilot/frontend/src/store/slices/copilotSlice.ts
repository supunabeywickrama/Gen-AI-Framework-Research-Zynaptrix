import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

interface TelemetryPoint {
  machineId: string;
  time: string;
  temperature: number;
  current: number;
  vibration: number;
}

export interface ChatMessage {
  role: 'agent' | 'user';
  content: string;
  images?: string[];
  machineId?: string;
}

export interface AnomalyRecord {
  id: number;
  machine_id: string;
  timestamp: string;
  type: string;
  score: number;
  sensor_data: string;
}

interface CopilotState {
  telemetry: TelemetryPoint[];
  // Keyed by anomaly ID (string) or 'general'
  chatHistory: Record<string, ChatMessage[]>;
  anomalyHistory: AnomalyRecord[];
  activeAnomaly: AnomalyRecord | null;
  systemState: 'NORMAL' | 'ANOMALY';
  anomalyScore: number;
  activeAgents: string[];
}

const initialState: CopilotState = {
  telemetry: [],
  chatHistory: {
    'general': [
      { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring multi-machine factory floor.' }
    ]
  },
  anomalyHistory: [],
  activeAnomaly: null,
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const fetchAnomalyHistory = createAsyncThunk(
  'copilot/fetchAnomalies',
  async (machineId: string) => {
    const response = await fetch(`${API_BASE}/api/machines/${machineId}/anomalies`);
    if (!response.ok) throw new Error('Failed to fetch anomaly history');
    return await response.json();
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
        content: 'Analyzing incident context...',
        targetId
    }));

    // Enhance payload with anomaly context if provided
    const body = {
        machine_id: payload.machine_id,
        user_query: payload.query,
        machine_state: payload.machine_state,
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
      // Initialize chat for this anomaly if it doesn't exist
      if (action.payload && !state.chatHistory[action.payload.id.toString()]) {
        state.chatHistory[action.payload.id.toString()] = [
            { role: 'agent', content: `🚨 Diagnostic session started for Incident #${action.payload.id}. Analysis of readings for ${action.payload.machine_id} is ready.` }
        ];
      }
    },
    setSystemState(state, action: PayloadAction<'NORMAL' | 'ANOMALY'>) {
      state.systemState = action.payload;
    },
    setAnomalyScore(state, action: PayloadAction<number>) {
      state.anomalyScore = action.payload;
    },
    setActiveAgents(state, action: PayloadAction<string[]>) {
      state.activeAgents = action.payload;
    }
  },
  extraReducers: (builder) => {
    builder.addCase(fetchAnomalyHistory.fulfilled, (state, action) => {
      state.anomalyHistory = action.payload;
    });
    builder.addCase(inquireCopilot.fulfilled, (state, action) => {
      const targetId = action.meta.arg.context_anomaly?.id.toString() || 'general';
      const history = state.chatHistory[targetId];
      if (history && history.length > 0) {
        const lastIdx = history.length - 1;
        const result = action.payload.graph_result;
        history[lastIdx] = {
            role: 'agent',
            content: result.final_execution_plan,
            images: result.retrieved_images,
            machineId: action.meta.arg.machine_id
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
  setActiveAgents
} = copilotSlice.actions;

export default copilotSlice.reducer;

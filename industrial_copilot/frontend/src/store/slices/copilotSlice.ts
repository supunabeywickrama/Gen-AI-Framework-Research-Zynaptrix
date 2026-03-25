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

interface CopilotState {
  telemetry: TelemetryPoint[];
  chatHistory: ChatMessage[];
  systemState: 'NORMAL' | 'ANOMALY';
  anomalyScore: number;
  activeAgents: string[];
}

const initialState: CopilotState = {
  telemetry: [],
  chatHistory: [
    { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring multi-machine factory floor.' }
  ],
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const inquireCopilot = createAsyncThunk(
  'copilot/inquire',
  async (payload: { machine_id: string; query: string; machine_state: string }, { dispatch }) => {
    // Add user message first
    dispatch(addChatMessage({ role: 'user', content: payload.query, machineId: payload.machine_id }));
    
    // Add a placeholder agent message
    dispatch(addChatMessage({ role: 'agent', content: 'Thinking...' }));
    const msgId = 1111; // We'd need a real ID system for updates, using index for now

    const response = await fetch(`${API_BASE}/api/copilot/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
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
    addChatMessage(state, action: PayloadAction<ChatMessage>) {
      state.chatHistory.push(action.payload);
    },
    updateChatMessage(state, action: PayloadAction<{ index: number; content: string; images?: string[] }>) {
      if (state.chatHistory[action.payload.index]) {
        state.chatHistory[action.payload.index].content = action.payload.content;
        if (action.payload.images) {
          state.chatHistory[action.payload.index].images = action.payload.images;
        }
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
    builder.addCase(inquireCopilot.fulfilled, (state, action) => {
      const lastIdx = state.chatHistory.length - 1;
      const result = action.payload.graph_result;
      state.chatHistory[lastIdx] = {
        role: 'agent',
        content: result.final_execution_plan,
        images: result.retrieved_images,
        machineId: action.meta.arg.machine_id
      };
    });
    builder.addCase(inquireCopilot.rejected, (state) => {
       const lastIdx = state.chatHistory.length - 1;
       state.chatHistory[lastIdx].content = "⚠️ Error communicating with Copilot backend.";
    });
  }
});

export const { 
  addTelemetry, 
  addChatMessage, 
  updateChatMessage,
  setSystemState, 
  setAnomalyScore,
  setActiveAgents
} = copilotSlice.actions;

export default copilotSlice.reducer;

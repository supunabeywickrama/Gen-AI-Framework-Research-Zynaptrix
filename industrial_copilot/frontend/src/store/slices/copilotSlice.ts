import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface TelemetryPoint {
  time: string;
  temperature: number;
  pressure: number;
  vibration: number;
}

export interface ChatMessage {
  role: 'agent' | 'user';
  content: string;
}

interface CopilotState {
  telemetry: TelemetryPoint[];
  chatHistory: ChatMessage[];
  systemState: 'NORMAL' | 'ANOMALY';
  anomalyScore: number;
  activeAgents: string[];
}

const initialState: CopilotState = {
  telemetry: [{ time: '10:00', temperature: 80, pressure: 40, vibration: 5 }],
  chatHistory: [
    { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring real-time sensor streams via InfluxDB.' }
  ],
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
};

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
    updateChatMessage(state, action: PayloadAction<{ index: number; content: string }>) {
      if (state.chatHistory[action.payload.index]) {
        state.chatHistory[action.payload.index].content = action.payload.content;
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

import { createSlice, PayloadAction } from '@reduxjs/toolkit';

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
  currentMachineId: string;
}

const initialState: CopilotState = {
  telemetry: [],
  chatHistory: [
    { role: 'agent', content: '🏭 Industrial Copilot initialized. Monitoring multi-machine factory floor.' }
  ],
  systemState: 'NORMAL',
  anomalyScore: 0.001,
  activeAgents: ['Sensor', 'Diagnostic', 'Strategy', 'Critic', 'RAG'],
  currentMachineId: 'PUMP-001'
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
    },
    setCurrentMachineId(state, action: PayloadAction<string>) {
      state.currentMachineId = action.payload;
    }
  },
});

export const { 
  addTelemetry, 
  addChatMessage, 
  updateChatMessage,
  setSystemState, 
  setAnomalyScore,
  setActiveAgents,
  setCurrentMachineId
} = copilotSlice.actions;

export default copilotSlice.reducer;

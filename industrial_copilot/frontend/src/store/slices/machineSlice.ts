import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

export interface Machine {
  machine_id: string;
  name: string;
  location: string;
  manual_id: string;
}

interface MachineState {
  machines: Machine[];
  currentMachineId: string;
  loading: boolean;
  error: string | null;
}

const initialState: MachineState = {
  machines: [],
  currentMachineId: 'PUMP-001',
  loading: false,
  error: null,
};

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const fetchMachines = createAsyncThunk('machines/fetchMachines', async () => {
    const response = await fetch(`${API_BASE}/machines`);
    if (!response.ok) throw new Error('Failed to fetch machines');
    return (await response.json()) as Machine[];
});

const machineSlice = createSlice({
  name: 'machines',
  initialState,
  reducers: {
    setCurrentMachineId(state, action: PayloadAction<string>) {
      state.currentMachineId = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMachines.pending, (state) => {
        state.loading = true;
      })
      .addCase(fetchMachines.fulfilled, (state, action) => {
        state.loading = false;
        state.machines = action.payload;
      })
      .addCase(fetchMachines.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Unknown error';
      });
  },
});

export const { setCurrentMachineId } = machineSlice.actions;
export default machineSlice.reducer;

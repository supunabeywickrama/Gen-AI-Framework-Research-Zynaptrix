"use client"
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  ChevronRight, 
  Database, 
  Gauge, 
  MessageSquare, 
  Play, 
  Send, 
  Server, 
  ShieldCheck, 
  Square, 
  Thermometer, 
  Vibrate as VibrateIcon,
  ArrowRight
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { 
  addTelemetry, 
  addChatMessage, 
  addAnomalyToHistory,
  setActiveAnomaly,
  setSystemState, 
  setAnomalyScore,
  fetchAnomalyHistory,
  inquireCopilot
} from '../store/slices/copilotSlice';
import { fetchMachines, setCurrentMachineId } from '../store/slices/machineSlice';
import { fetchSimulatorStatus, startSimulator, stopSimulator } from '../store/slices/simulatorSlice';

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  
  // Redux-driven State
  const { telemetry, chatHistory, anomalyHistory, activeAnomaly, systemState, anomalyScore } = useSelector((state: RootState) => state.copilot);
  const { machines, currentMachineId } = useSelector((state: RootState) => state.machines);
  const { activeSimulators } = useSelector((state: RootState) => state.simulator);
  
  // Local transient UI states
  const [query, setQuery] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Filter telemetry points for the current machine
  const filteredTelemetry = telemetry.filter(t => t.machineId === currentMachineId);
  const latestReading = filteredTelemetry[filteredTelemetry.length - 1] || { temperature: 0, current: 0, vibration: 0 };
  
  const isSimulating = activeSimulators.includes(currentMachineId);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    dispatch(fetchMachines());
    dispatch(fetchSimulatorStatus());
  }, [dispatch]);

  useEffect(() => {
    if (currentMachineId) {
      dispatch(fetchAnomalyHistory(currentMachineId));
    }
  }, [currentMachineId, dispatch]);

  // WebSocket Telemetry Stream
  useEffect(() => {
    let ws: WebSocket;
    const connectWS = () => {
      const base = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const wsUrl = base.replace('http', 'ws');
      ws = new WebSocket(`${wsUrl}/ws/telemetry`);
      
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === "telemetry") {
            const data = parsed.data;
            dispatch(addTelemetry({
              machineId: data.machine_id || 'PUMP-001',
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              temperature: data.temperature,
              current: data.motor_current,
              vibration: data.vibration
            }));
          } else if (parsed.type === "anomaly_alert") {
            const result = parsed.data;
            dispatch(setSystemState('ANOMALY'));
            dispatch(setAnomalyScore(result.anomaly_score || 0.99));
            dispatch(addAnomalyToHistory({
                id: result.id,
                machine_id: result.machine_id,
                timestamp: new Date().toLocaleTimeString(),
                type: result.machine_state,
                score: Math.round(result.anomaly_score * 100),
                sensor_data: JSON.stringify(result.recent_readings || {})
            }));
          }
        } catch (err) { console.error(err); }
      };
      ws.onclose = () => setTimeout(connectWS, 3000);
    };
    connectWS();
    return () => ws?.close();
  }, [dispatch]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isChatOpen]);

  const handleManualInquiry = (customQuery?: string) => {
    const finalQuery = customQuery || query;
    if (!finalQuery) return;
    dispatch(inquireCopilot({
      machine_id: currentMachineId,
      query: finalQuery,
      machine_state: activeAnomaly ? activeAnomaly.type : "manual_inquiry_general",
      context_anomaly: activeAnomaly || undefined
    }));
    setQuery('');
  };

  const toggleSimulation = async () => {
    if (isSimulating) await dispatch(stopSimulator(currentMachineId));
    else await dispatch(startSimulator(currentMachineId));
    dispatch(fetchSimulatorStatus());
  };

  const diagnosticSuggestions = [
    "Likely causes for this fault?",
    "Technical repair checklist",
    "Check signal-line integrity",
    "Analyze ADC voltage drift",
    "Inspect terminals for oxidation"
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans selection:bg-blue-500/30 overflow-hidden flex flex-col">
      
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-6 bg-slate-900/40 backdrop-blur-md shadow-xl rounded-2xl p-6">
        <div>
          <h1 className="text-3xl font-black bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-3">
            <Server className="text-blue-500" />
            Industrial Copilot Hub
          </h1>
          <div className="flex items-center gap-2 mt-1">
             <div className={`h-2 w-2 rounded-full ${isSimulating ? 'bg-emerald-500 animate-pulse' : 'bg-slate-600'}`}></div>
             <p className="text-slate-400 text-[10px] font-bold tracking-widest uppercase opacity-70">
               {isSimulating ? 'Live Telemetry Active' : 'Simulator Standby'}
             </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex bg-slate-800/50 rounded-xl p-1 border border-slate-700/50">
            {machines.map(m => (
              <button
                key={m.machine_id}
                onClick={() => dispatch(setCurrentMachineId(m.machine_id))}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${currentMachineId === m.machine_id ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {m.name}
              </button>
            ))}
          </div>
          <button
              onClick={toggleSimulation}
              className={`px-5 py-2.5 rounded-xl font-black text-[10px] uppercase tracking-widest transition-all flex items-center gap-2 shadow-2xl ${isSimulating ? 'bg-red-500/10 text-red-500 border border-red-500/30' : 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/30'}`}
            >
              {isSimulating ? <Square size={12} fill="currentColor" /> : <Play size={12} fill="currentColor" />}
              {isSimulating ? "Stop" : "Start"}
          </button>
        </div>
      </header>

      {/* Main Grid: Left (Sensor Lines) | Right (Detected Anomalies) */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 flex-1 overflow-hidden">
        
        {/* Left Aspect: Sensor Telemetry (2/3 width) */}
        <div className="xl:col-span-2 flex flex-col gap-6 overflow-hidden">
          
          <div className="grid grid-cols-3 gap-4">
             {[
               { label: 'Temperature', val: latestReading.temperature.toFixed(1), unit: '°C', color: 'blue', icon: Thermometer },
               { label: 'Motor Load', val: latestReading.current.toFixed(2), unit: 'A', color: 'emerald', icon: Gauge },
               { label: 'Vibration', val: latestReading.vibration.toFixed(2), unit: 'mm/s', color: 'amber', icon: VibrateIcon },
             ].map((kpi, i) => (
               <div key={i} className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl relative overflow-hidden group">
                  <div className={`absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity text-${kpi.color}-500`}>
                     <kpi.icon size={60} />
                  </div>
                  <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">{kpi.label}</h3>
                  <p className="text-3xl font-black text-white">{kpi.val} <span className="text-sm text-slate-600 font-light">{kpi.unit}</span></p>
               </div>
             ))}
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-6">
               <h2 className="text-xl font-black flex items-center gap-3 text-white">
                 <Activity className="text-blue-500" /> Real-Time Sensor Stream
               </h2>
               <div className="flex items-center gap-4 text-[10px] font-bold text-slate-500">
                  <span className="flex items-center gap-2"><div className="h-1.5 w-4 bg-blue-500 rounded"></div> Temp</span>
                  <span className="flex items-center gap-2"><div className="h-1.5 w-4 bg-emerald-500 rounded"></div> Load</span>
                  <span className="flex items-center gap-2"><div className="h-1.5 w-4 bg-amber-500 rounded"></div> Vibr</span>
               </div>
            </div>
            <div className="flex-1 w-full min-h-[300px]">
              {isMounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={filteredTelemetry}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#475569" tick={{fill: '#475569', fontSize: 10, fontWeight: 700}} hide />
                    <YAxis stroke="#475569" tick={{fill: '#475569', fontSize: 10, fontWeight: 700}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                    />
                    <Line type="monotone" dataKey="temperature" stroke="#3b82f6" strokeWidth={3} dot={false} animationDuration={300} />
                    <Line type="monotone" dataKey="current" stroke="#10b981" strokeWidth={3} dot={false} animationDuration={300} />
                    <Line type="monotone" dataKey="vibration" stroke="#f59e0b" strokeWidth={3} dot={false} animationDuration={300} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Right Aspect: Anomaly Archive (1/3 width) */}
        <div className="flex flex-col gap-6 overflow-hidden">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl flex flex-col h-full overflow-hidden">
            <h2 className="text-lg font-black text-white mb-6 flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Database className="text-blue-400" size={20} /> Incident Registry
              </span>
              <div className={`px-3 py-1 rounded-full text-[10px] font-black border ${systemState === 'NORMAL' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border-red-500/30 text-red-500'}`}>
                {systemState}
              </div>
            </h2>

            <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-800">
               {anomalyHistory.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center opacity-20 py-20 text-center">
                    <CheckCircle size={40} className="mb-4" />
                    <p className="text-xs font-bold uppercase tracking-widest">No Anomalies Logged</p>
                 </div>
               ) : (
                 anomalyHistory.map((item) => (
                   <button
                    key={item.id}
                    onClick={() => {
                        dispatch(setActiveAnomaly(item));
                        setIsChatOpen(true);
                    }}
                    className={`w-full p-4 rounded-2xl border transition-all text-left relative group ${activeAnomaly?.id === item.id ? 'bg-blue-600/10 border-blue-500' : 'bg-slate-950/50 border-slate-800/50 hover:bg-slate-900'}`}
                   >
                     <div className="flex justify-between items-start mb-2">
                        <span className={`text-[9px] font-black px-2 py-0.5 rounded uppercase tracking-[0.1em] ${item.type.includes('fault') ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'}`}>
                           {item.type.replace('machine_', '')}
                        </span>
                        <span className="text-[9px] text-slate-500 font-mono italic">{item.timestamp}</span>
                     </div>
                     <p className="text-sm font-bold text-slate-200">Industrial Incident #{item.id}</p>
                     <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-tighter">Variance Score: {item.score}%</p>
                     <div className="absolute right-4 bottom-4 opacity-0 group-hover:opacity-100 transition-all text-blue-400">
                        <ArrowRight size={16} />
                     </div>
                   </button>
                 ))
               )}
            </div>
          </div>
        </div>
      </div>

      {/* Floating Chat Trigger */}
      <button 
        onClick={() => setIsChatOpen(true)}
        className="fixed bottom-8 right-8 h-16 w-16 bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-2xl shadow-blue-500/40 flex items-center justify-center transition-all hover:scale-110 active:scale-95 group z-40"
      >
        <MessageSquare size={24} className="group-hover:rotate-12 transition-transform" />
        {activeAnomaly && (
            <div className="absolute -top-1 -right-1 h-5 w-5 bg-red-500 rounded-full border-2 border-slate-950 flex items-center justify-center text-[10px] font-black">1</div>
        )}
      </button>

      {/* Chat Modal / Pop-up */}
      {isChatOpen && (
        <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-md z-50 flex items-center justify-center p-4 animate-in fade-in duration-300">
          <div className="bg-slate-900 border border-slate-800 w-full max-w-2xl h-[85vh] rounded-[2.5rem] shadow-4xl flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-500">
             
             {/* Modal Header */}
             <div className="p-8 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
                <div>
                   <h2 className="text-2xl font-black flex items-center gap-3 text-white">
                      <ShieldCheck className="text-blue-500" /> Diagnostic Copilot
                   </h2>
                   {activeAnomaly ? (
                     <p className="text-blue-400 text-xs font-bold uppercase tracking-widest mt-1">Investigating Incident #{activeAnomaly.id}</p>
                   ) : (
                     <p className="text-slate-500 text-xs font-bold uppercase tracking-widest mt-1">General Inquiry Mode</p>
                   )}
                </div>
                <button 
                    onClick={() => setIsChatOpen(false)}
                    className="p-3 bg-slate-800 hover:bg-slate-700 rounded-2xl text-slate-400 hover:text-white transition-all"
                >
                    <Square size={20} />
                </button>
             </div>

             {/* Modal Chat Body */}
             <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] scrollbar-thin scrollbar-thumb-slate-700">
                {chatHistory.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        <div className={`max-w-[85%] p-6 rounded-3xl shadow-2xl ${msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-slate-800 text-slate-100 border border-slate-700 rounded-tl-none'}`}>
                            <div className="text-sm leading-relaxed space-y-4 markdown-content">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {msg.content
                                        .replace(/\[SUGGESTION:.*?\]/gi, '') // Hide suggestions from main text
                                        .replace(/\[IMAGE[_\s-]?(\d+)\]/gi, (match, id) => {
                                            const url = msg.images?.[parseInt(id)];
                                            return url ? `![Technical Diagram ${id}](${url})` : match;
                                        })
                                    }
                                </ReactMarkdown>
                                
                                {/* Dynamic AI Follow-up Suggestions */}
                                {msg.role === 'agent' && msg.content.includes('[SUGGESTION:') && (
                                    <div className="mt-6 pt-4 border-t border-slate-700/50 flex flex-wrap gap-2">
                                        {Array.from(msg.content.matchAll(/\[SUGGESTION:\s*(.*?)\]/gi)).map((match, si) => (
                                            <button 
                                                key={si} 
                                                onClick={() => handleManualInquiry(match[1])}
                                                className="text-[10px] uppercase font-black bg-blue-600/20 text-blue-400 px-3 py-1.5 rounded-lg border border-blue-500/30 hover:bg-blue-600 hover:text-white transition-all"
                                            >
                                                {match[1]}
                                            </button>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                ))}
                <div ref={chatEndRef} />
             </div>

             {/* Modal Footer */}
             <div className="p-8 bg-slate-950/80 border-t border-slate-800 space-y-6">
                <div className="flex flex-wrap gap-2">
                    {diagnosticSuggestions.map((s, i) => (
                        <button key={i} onClick={() => handleManualInquiry(s)} className="text-[10px] font-bold bg-slate-800 text-slate-300 px-4 py-2 rounded-xl hover:bg-blue-600 hover:text-white transition-all border border-slate-700">
                            {s}
                        </button>
                    ))}
                </div>
                <div className="flex items-center gap-4">
                    <input 
                        className="flex-1 bg-slate-900 border border-slate-800 rounded-2xl px-6 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-200"
                        placeholder="Type diagnostic query..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleManualInquiry()}
                    />
                    <button onClick={() => handleManualInquiry()} className="bg-blue-600 p-4 rounded-2xl hover:bg-blue-500 transition-all shadow-xl">
                        <Send size={24} />
                    </button>
                </div>
             </div>
          </div>
        </div>
      )}
    </div>
  );
}

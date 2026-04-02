"use client"
import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
// Step-by-step procedure flow is now inline in chat
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
  ShieldAlert,
  ShieldCheck, 
  Square, 
  Thermometer, 
  Vibrate as VibrateIcon,
  ArrowRight,
  Maximize2,
  Minimize2,
  Wrench,
  MessageCircle
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
  fetchChatHistory,
  resolveAnomaly,
  inquireCopilot,
  respondToStep
} from '../store/slices/copilotSlice';
import { fetchMachines, setCurrentMachineId, fetchMachineConfig } from '../store/slices/machineSlice';
import { fetchSimulatorStatus, startSimulator, stopSimulator } from '../store/slices/simulatorSlice';

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  
  // Redux-driven State
  const { telemetry, chatHistory, anomalyHistory, activeAnomaly, systemState, anomalyScore, loadingHistory, activeProcedure } = useSelector((state: RootState) => state.copilot);
  const { machines, currentMachineId, machineConfigs } = useSelector((state: RootState) => state.machines);
  const { activeSimulators } = useSelector((state: RootState) => state.simulator);
  
  // Local transient UI states
  const [query, setQuery] = useState('');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatMaximized, setIsChatMaximized] = useState(false);
  const [isResolveModalOpen, setIsResolveModalOpen] = useState(false);
  const [operatorFix, setOperatorFix] = useState('');
  const [stepComments, setStepComments] = useState<Record<string, string>>({});
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Filter telemetry points for the current machine
  const filteredTelemetry = telemetry.filter(t => t.machineId === currentMachineId);
  const latestReading = filteredTelemetry[filteredTelemetry.length - 1] || {};
  
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
      dispatch(fetchMachineConfig(currentMachineId));
    }
  }, [currentMachineId, dispatch]);

  // Load chat history and trigger initial diagnosis when an anomaly is selected
  useEffect(() => {
    if (activeAnomaly) {
      const anomalyIdStr = activeAnomaly.id.toString();
      dispatch(fetchChatHistory(activeAnomaly.id)).then((action: any) => {
        // action.payload is { anomalyId, messages }
        const history = action.payload?.messages;
        if (!history || history.length === 0) {
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
              ...data,
              machineId: data.machine_id || 'PUMP-001',
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
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
                sensor_data: JSON.stringify(result.recent_readings || {}),
                resolved: false
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
    if (!finalQuery || activeAnomaly?.resolved) return;
    dispatch(inquireCopilot({
      machine_id: currentMachineId,
      query: finalQuery,
      machine_state: activeAnomaly ? activeAnomaly.type : "manual_inquiry_general",
      context_anomaly: activeAnomaly || undefined
    }));
    setQuery('');
  };

  const handleResolve = async () => {
    if (!activeAnomaly || !operatorFix) return;
    await dispatch(resolveAnomaly({ 
        anomalyId: activeAnomaly.id, 
        operator_fix: operatorFix 
    }));
    setIsResolveModalOpen(false);
    setOperatorFix('');
  };

  const handleStepResponse = (stepId: string, status: string) => {
    if (!activeAnomaly) return;
    const targetId = activeAnomaly.id.toString();
    const comment = stepComments[stepId] || undefined;
    dispatch(respondToStep({ targetId, stepId, status, comment }));
    // Clear the comment for this step
    setStepComments(prev => { const n = {...prev}; delete n[stepId]; return n; });
  };

  // Helper: render step text with inline images
  const renderStepContent = (text: string, images?: string[]) => {
    const parts = text.split(/(\[IMAGE[_\s-]?\d+\])/gi);
    return parts.map((part, i) => {
      const match = part.match(/\[IMAGE[_\s-]?(\d+)\]/i);
      if (match && images) {
        const imgUrl = images[parseInt(match[1])];
        if (imgUrl) {
          return <img key={i} src={imgUrl} alt="Technical diagram" className="w-full max-w-lg h-auto rounded-xl border border-slate-700 my-3" />;
        }
      }
      return <span key={i}>{part}</span>;
    });
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
             {(() => {
                const config = machineConfigs[currentMachineId];
                const sensorKeys = Object.keys(latestReading).filter(k => !['machineId', 'time', 'machine_id', 'state'].includes(k));
                
                // Use keys from latest telemetry, or if empty, use keys from machine config, or fallback to defaults
                let displaySensors = sensorKeys.length > 0 ? sensorKeys : ['temperature', 'current', 'vibration'];
                // machineConfigs stores a string[] of sensor IDs
                if (sensorKeys.length === 0 && config && Array.isArray(config)) {
                   displaySensors = config;
                }

                const colors = ['blue', 'emerald', 'amber', 'purple', 'rose', 'cyan'];
                const icons = [Thermometer, Gauge, VibrateIcon, Activity, Database, Server];
                return displaySensors.map((key, i) => {
                  const val = latestReading[key] !== undefined ? Number(latestReading[key]).toFixed(2) : '0.00';
                  const color = colors[i % colors.length];
                  const Icon = icons[i % icons.length];
                  return (
                   <div key={key} className={`bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl relative overflow-hidden group`}>
                      <div className={`absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity text-${color}-500`}>
                         <Icon size={60} />
                      </div>
                      <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">{key.replace(/_/g, ' ')}</h3>
                      <p className="text-3xl font-black text-white">{val} <span className="text-sm text-slate-600 font-light">units</span></p>
                   </div>
                  );
                });
             })()}
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl relative overflow-hidden flex-1 flex flex-col">
            <div className="flex justify-between items-center mb-6">
               <h2 className="text-xl font-black flex items-center gap-3 text-white">
                 <Activity className="text-blue-500" /> Real-Time Sensor Stream
               </h2>
               <div className="flex items-center gap-4 text-[10px] font-bold text-slate-500">
                  {(() => {
                    const sensorKeys = Object.keys(latestReading).filter(k => !['machineId', 'time', 'machine_id', 'state'].includes(k));
                    const displaySensors = sensorKeys.length > 0 ? sensorKeys : ['temperature', 'current', 'vibration'];
                    const colors = ['bg-blue-500', 'bg-emerald-500', 'bg-amber-500', 'bg-purple-500', 'bg-rose-500', 'bg-cyan-500'];
                    return displaySensors.map((key, i) => (
                      <span key={key} className="flex items-center gap-2">
                        <div className={`h-1.5 w-4 ${colors[i % colors.length]} rounded`}></div>
                        {key.substring(0, 4).toUpperCase()}
                      </span>
                    ));
                  })()}
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
                    {(() => {
                       const config = machineConfigs[currentMachineId];
                       const sensorKeys = Object.keys(latestReading).filter(k => !['machineId', 'time', 'machine_id', 'state'].includes(k));
                       let displaySensors = sensorKeys.length > 0 ? sensorKeys : ['temperature', 'current', 'vibration'];
                       if (sensorKeys.length === 0 && config && Array.isArray(config)) {
                          displaySensors = config;
                       }
                       const colors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#f43f5e', '#06b6d4'];
                       return displaySensors.map((key, i) => (
                         <Line key={key} type="monotone" dataKey={key} stroke={colors[i % colors.length]} strokeWidth={3} dot={false} animationDuration={300} />
                       ));
                    })()}
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
                        <div className="flex items-center gap-2">
                            {item.resolved && <CheckCircle size={10} className="text-emerald-500" />}
                            <span className="text-[9px] text-slate-500 font-mono italic">{item.timestamp}</span>
                        </div>
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
        <div className={`fixed inset-0 bg-slate-950/80 backdrop-blur-md z-50 flex items-center justify-center animate-in fade-in duration-300 ${isChatMaximized ? 'p-0' : 'p-4'}`}>
          <div className={`bg-slate-900 border border-slate-800 shadow-4xl flex flex-col overflow-hidden animate-in zoom-in-95 slide-in-from-bottom-10 duration-500 transition-all ${isChatMaximized ? 'w-full h-full rounded-none' : 'w-full max-w-4xl h-[85vh] rounded-[2.5rem]'}`}>
             
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
                 <div className="flex items-center gap-2">
                    {activeAnomaly && !activeAnomaly.resolved && (
                        <button 
                            onClick={() => setIsResolveModalOpen(true)}
                            className="flex items-center gap-2 px-4 py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-2xl text-[10px] font-black uppercase tracking-widest transition-all shadow-lg shadow-emerald-500/20"
                        >
                            <CheckCircle size={14} /> Complete Task
                        </button>
                    )}
                    <button 
                        onClick={() => setIsChatMaximized(!isChatMaximized)}
                        className="p-3 bg-slate-800 hover:bg-slate-700 rounded-2xl text-slate-400 hover:text-white transition-all"
                        title={isChatMaximized ? "Minimize" : "Maximize"}
                    >
                        {isChatMaximized ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
                    </button>
                    <button 
                        onClick={() => setIsChatOpen(false)}
                        className="p-3 bg-red-600/20 hover:bg-red-600 rounded-2xl text-red-400 hover:text-white transition-all"
                    >
                        <Square size={20} />
                    </button>
                </div>
             </div>

             {/* Modal Chat Body */}
             <div className="flex-1 overflow-y-auto p-8 space-y-6 bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] scrollbar-thin scrollbar-thumb-slate-700">
                {loadingHistory[activeAnomaly?.id.toString() || ''] && (!chatHistory[activeAnomaly?.id.toString() || ''] || chatHistory[activeAnomaly?.id.toString() || ''].length === 0) ? (
                    <div className="h-full flex flex-col items-center justify-center opacity-50 space-y-4">
                        <div className="h-12 w-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        <p className="text-xs font-bold uppercase tracking-widest">Restoring Context...</p>
                    </div>
                ) : (
                    (chatHistory[activeAnomaly?.id.toString() || 'general'] || []).map((msg, i) => {
                        const hasSuggestion = msg.role === 'agent' && msg.content.includes('[SUGGESTION:');
                        const displayContent = msg.content
                            .replace(/\[PROCEDURE_START\][\s\S]*?\[PROCEDURE_END\]/gi, '')
                            .replace(/\[ACTION:\s*(.*?)\]/gi, '')
                            .replace(/\[SUGGESTION:.*?\]/gi, '')
                            .trim();

                        // === PHASE HEADER ===
                        if (msg.type === 'phase_header') {
                          return (
                            <div key={i} className="flex justify-center my-6">
                              <div className="bg-gradient-to-r from-slate-800 via-slate-700 to-slate-800 text-white px-8 py-4 rounded-2xl border border-slate-600 shadow-xl text-center max-w-lg">
                                <div className="text-sm font-black leading-relaxed">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                </div>
                              </div>
                            </div>
                          );
                        }

                        // === INTERACTIVE STEP CARD ===
                        if (msg.type === 'step' && msg.stepData) {
                          const sd = msg.stepData;
                          const isSafety = sd.phaseType === 'safety';
                          const isResponded = !!msg.stepResponse;

                          return (
                            <div key={i} className="flex justify-start">
                              <div className={`max-w-[90%] w-full rounded-3xl shadow-2xl border overflow-hidden transition-all duration-500 ${
                                isResponded 
                                  ? 'bg-slate-800/50 border-slate-700/50 opacity-70' 
                                  : isSafety 
                                    ? 'bg-gradient-to-br from-amber-950/40 to-slate-800 border-amber-500/40' 
                                    : 'bg-gradient-to-br from-blue-950/40 to-slate-800 border-blue-500/40'
                              }`}>
                                <div className={`px-6 py-3 flex items-center justify-between ${
                                  isSafety ? 'bg-amber-500/10 border-b border-amber-500/20' : 'bg-blue-500/10 border-b border-blue-500/20'
                                }`}>
                                  <div className="flex items-center gap-2">
                                    {isSafety ? <ShieldAlert size={16} className="text-amber-400" /> : <Wrench size={16} className="text-blue-400" />}
                                    <span className={`text-[10px] font-black uppercase tracking-widest ${isSafety ? 'text-amber-400' : 'text-blue-400'}`}>
                                      {sd.subphaseTitle}
                                    </span>
                                  </div>
                                  <span className="text-[10px] font-bold text-slate-500 bg-slate-800 px-3 py-1 rounded-full">
                                    Step {sd.stepIndex + 1} of {sd.totalSteps}
                                  </span>
                                </div>

                                <div className="p-6">
                                  <p className="text-white text-base font-medium leading-relaxed">
                                    {renderStepContent(msg.content, msg.images)}
                                  </p>

                                  {sd.critical && !isResponded && (
                                    <div className="inline-flex items-center gap-1.5 bg-red-500/15 text-red-400 text-[10px] font-black uppercase tracking-widest px-3 py-1.5 rounded-lg border border-red-500/30 mt-3">
                                      <AlertTriangle size={12} /> Safety Critical
                                    </div>
                                  )}

                                  {isResponded && msg.stepResponse && (
                                    <div className={`mt-4 flex items-center gap-2 text-sm font-bold ${
                                      msg.stepResponse.status === 'done' ? 'text-emerald-400' : 'text-red-400'
                                    }`}>
                                      <CheckCircle size={16} />
                                      <span className="capitalize">{msg.stepResponse.status.replace('_', ' ')}</span>
                                      {msg.stepResponse.comment && <span className="text-slate-400 font-normal ml-2">&mdash; &quot;{msg.stepResponse.comment}&quot;</span>}
                                    </div>
                                  )}

                                  {!isResponded && (
                                    <div className="mt-6 space-y-4">
                                      {isSafety ? (
                                        <div className="flex flex-wrap gap-3">
                                          <button onClick={() => handleStepResponse(sd.stepId, 'done')} className="flex-1 min-w-[130px] py-3 px-4 bg-emerald-600/20 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-500/30 rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2">
                                            <CheckCircle size={14} /> I have done
                                          </button>
                                          <button onClick={() => handleStepResponse(sd.stepId, 'havent_done')} className="flex-1 min-w-[130px] py-3 px-4 bg-yellow-600/20 hover:bg-yellow-600 text-yellow-400 hover:text-white border border-yellow-500/30 rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2">
                                            ⏳ Haven&apos;t done
                                          </button>
                                          <button onClick={() => handleStepResponse(sd.stepId, 'cant_do')} className="flex-1 min-w-[130px] py-3 px-4 bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white border border-red-500/30 rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2">
                                            <AlertTriangle size={14} /> Can&apos;t do it
                                          </button>
                                        </div>
                                      ) : (
                                        <div className="space-y-3">
                                          <input
                                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            placeholder="Add a comment (optional)..."
                                            value={stepComments[sd.stepId] || ''}
                                            onChange={(e) => setStepComments(prev => ({...prev, [sd.stepId]: e.target.value}))}
                                          />
                                          <div className="flex gap-3">
                                            <button onClick={() => handleStepResponse(sd.stepId, 'done')} className="flex-1 py-3 px-6 bg-emerald-600/20 hover:bg-emerald-600 text-emerald-400 hover:text-white border border-emerald-500/30 rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2">
                                              <CheckCircle size={14} /> Done
                                            </button>
                                            <button onClick={() => handleStepResponse(sd.stepId, 'undone')} className="flex-1 py-3 px-6 bg-red-600/20 hover:bg-red-600 text-red-400 hover:text-white border border-red-500/30 rounded-xl text-xs font-black uppercase tracking-wider transition-all flex items-center justify-center gap-2">
                                              <AlertTriangle size={14} /> Undone
                                            </button>
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          );
                        }

                        // === PROCEDURE COMPLETE ===
                        if (msg.type === 'procedure_complete') {
                          return (
                            <div key={i} className="flex justify-center my-6">
                              <div className="bg-gradient-to-r from-emerald-900/30 to-emerald-800/20 border border-emerald-500/30 text-white px-10 py-8 rounded-3xl shadow-2xl text-center max-w-lg">
                                <div className="text-5xl mb-4">🎉</div>
                                <div className="text-lg font-black mb-2">All Steps Completed!</div>
                                <p className="text-sm text-slate-400">Click <span className="text-emerald-400 font-bold">Complete Task</span> above to finalize.</p>
                              </div>
                            </div>
                          );
                        }

                        // === USER MESSAGE ===
                        if (msg.role === 'user') {
                          return (
                            <div key={i} className="flex justify-end">
                              <div className="max-w-[75%] bg-blue-600 text-white rounded-3xl rounded-br-none p-5 shadow-xl">
                                <p className="text-sm leading-relaxed whitespace-pre-line">{msg.content}</p>
                              </div>
                            </div>
                          );
                        }

                        // === AGENT TEXT (summary / regular) ===
                        return (
                          <div key={i} className="flex justify-start">
                            <div className="max-w-[85%] bg-slate-800 text-slate-100 border border-slate-700 rounded-3xl rounded-tl-none p-6 shadow-2xl">
                              {displayContent && (
                                <div className="text-sm leading-relaxed space-y-3 markdown-content">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ img: ({node, ...props}) => (<img {...props} className="max-w-md max-h-80 object-contain rounded-xl border border-slate-600 my-3" />) }}>
                                    {displayContent}
                                  </ReactMarkdown>
                                </div>
                              )}
                              {hasSuggestion && !activeAnomaly?.resolved && (
                                <div className="mt-6 pt-5 border-t border-slate-700/50">
                                  <button onClick={() => handleManualInquiry("Generate full step-by-step repair procedure")} className="w-full py-4 px-6 bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-black text-sm uppercase tracking-widest rounded-2xl flex items-center justify-center gap-3 transition-all duration-300 shadow-xl shadow-blue-500/25 hover:shadow-blue-500/40 hover:scale-[1.02] active:scale-95">
                                    <ShieldCheck size={20} /> 🔧 Start Guided Repair Procedure <ArrowRight size={18} />
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                    })
                )}

                <div ref={chatEndRef} />
             </div>

             {/* Modal Footer */}
             <div className="p-8 bg-slate-950/80 border-t border-slate-800 space-y-6">
                {!activeAnomaly?.resolved && (
                    <div className="flex flex-wrap gap-2">
                        {diagnosticSuggestions.map((s, i) => (
                            <button key={i} onClick={() => handleManualInquiry(s)} className="text-[10px] font-bold bg-slate-800 text-slate-300 px-4 py-2 rounded-xl hover:bg-blue-600 hover:text-white transition-all border border-slate-700">
                                {s}
                            </button>
                        ))}
                    </div>
                )}
                <div className="flex items-center gap-4">
                    <input 
                        className={`flex-1 bg-slate-900 border border-slate-800 rounded-2xl px-6 py-4 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 text-slate-200 ${activeAnomaly?.resolved ? 'opacity-50 cursor-not-allowed' : ''}`}
                        placeholder={activeAnomaly?.resolved ? "Incident resolved (Archived)" : "Type diagnostic query..."}
                        value={query}
                        readOnly={activeAnomaly?.resolved}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handleManualInquiry()}
                    />
                    <button 
                        onClick={() => handleManualInquiry()} 
                        disabled={activeAnomaly?.resolved}
                        className={`bg-blue-600 p-4 rounded-2xl hover:bg-blue-500 transition-all shadow-xl ${activeAnomaly?.resolved ? 'opacity-30 cursor-not-allowed' : ''}`}
                    >
                        <Send size={24} />
                    </button>
                </div>
             </div>
          </div>
        </div>
      )}

      {/* Incident Resolution Modal */}
      {isResolveModalOpen && (
        <div className="fixed inset-0 bg-slate-950/90 backdrop-blur-xl z-[60] flex items-center justify-center p-4">
           <div className="bg-slate-900 border border-slate-800 rounded-[3rem] p-10 max-w-lg w-full shadow-4xl animate-in zoom-in-95 duration-300">
              <div className="h-16 w-16 bg-emerald-600/20 text-emerald-500 rounded-3xl flex items-center justify-center mb-6">
                 <CheckCircle size={32} />
              </div>
              <h2 className="text-2xl font-black text-white mb-2">Finalize Diagnostic</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-8">
                Briefly describe the **actual manual actions** taken on the machine. This will be vectorized into the AI knowledge base for future troubleshooting.
              </p>
              
              <textarea 
                className="w-full bg-slate-950 border border-slate-800 rounded-3xl p-6 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-emerald-500 min-h-[150px] mb-8"
                placeholder="e.g. Manually bled the hydraulic lines and replaced the leaking filter at ACV-102. Thermal integrity restored."
                value={operatorFix}
                onChange={(e) => setOperatorFix(e.target.value)}
              />

              <div className="flex gap-4">
                 <button 
                    onClick={() => setIsResolveModalOpen(false)}
                    className="flex-1 px-6 py-4 rounded-2xl bg-slate-800 text-slate-400 font-bold hover:text-white transition-all text-xs uppercase tracking-widest"
                 >
                    Cancel
                 </button>
                 <button 
                    onClick={handleResolve}
                    disabled={!operatorFix.trim()}
                    className="flex-[2] px-6 py-4 rounded-2xl bg-emerald-600 text-white font-black hover:bg-emerald-500 transition-all text-xs uppercase tracking-widest shadow-xl shadow-emerald-500/20 disabled:opacity-30"
                 >
                    Archive Incident
                 </button>
              </div>
           </div>
        </div>
      )}
    </div>
  );
}

"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, ShieldCheck, Server, MessageSquare, Send, Thermometer, Gauge, Activity as VibrateIcon, UploadCloud, FileText, CheckCircle, Loader2, Play, Square } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { addChatMessage, addTelemetry, setSystemState, setAnomalyScore, inquireCopilot } from '../store/slices/copilotSlice';
import { fetchMachines, setCurrentMachineId } from '../store/slices/machineSlice';
import { fetchSimulatorStatus, startSimulator, stopSimulator } from '../store/slices/simulatorSlice';
import { uploadManual, clearUploadStatus } from '../store/slices/ingestionSlice';

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  
  // Redux-driven State
  const { telemetry, chatHistory, systemState, anomalyScore } = useSelector((state: RootState) => state.copilot);
  const { machines, currentMachineId, loading: machinesLoading } = useSelector((state: RootState) => state.machines);
  const { activeSimulators } = useSelector((state: RootState) => state.simulator);
  const { isUploading, uploadStatus } = useSelector((state: RootState) => state.ingestion);
  
  const [query, setQuery] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Filter telemetry points for the current machine
  const filteredTelemetry = telemetry.filter(t => t.machineId === currentMachineId);
  const latestReading = filteredTelemetry[filteredTelemetry.length - 1] || { temperature: 0, current: 0, vibration: 0 };
  
  // Local transient UI states
  const [manualId, setManualId] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  
  const isSimulating = activeSimulators.includes(currentMachineId);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
    // Initial Bootstrap
    dispatch(fetchMachines());
    dispatch(fetchSimulatorStatus());
  }, [dispatch]);

  // WebSocket Telemetry Stream (Centralized Dispatcher)
  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

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
            
            dispatch(addChatMessage({ 
              role: 'agent', 
              content: result.final_execution_plan || "Critical Anomaly Detected.",
              images: result.retrieved_images || [],
              machineId: result.machine_id
            }));
          }
        } catch (err) {
          console.error("Telemetry parsing error:", err);
        }
      };

      ws.onclose = () => {
        reconnectTimer = setTimeout(connectWS, 3000);
      };
      
      ws.onerror = () => {
        ws.close();
      };
    };

    connectWS();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [dispatch]);

  // UI Helpers
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  // Thunk-based Interactions
  const handleManualInquiry = () => {
    if (!query) return;
    dispatch(inquireCopilot({
      machine_id: currentMachineId,
      query,
      machine_state: systemState === "NORMAL" ? "manual_inquiry_normal" : "anomaly_investigation",
    }));
    setQuery('');
  };

  const toggleSimulation = async () => {
    if (isSimulating) {
      await dispatch(stopSimulator(currentMachineId));
    } else {
      await dispatch(startSimulator(currentMachineId));
    }
    // Refresh global status after toggle
    dispatch(fetchSimulatorStatus());
  };

  const handleUpload = () => {
    if (!uploadFile || !manualId) return;
    dispatch(uploadManual({ manualId, file: uploadFile })).then((res) => {
      if (res.meta.requestStatus === 'fulfilled') {
        setTimeout(() => {
          dispatch(clearUploadStatus());
          setManualId("");
          setUploadFile(null);
        }, 3000);
      }
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans selection:bg-blue-500/30 overflow-x-hidden">
      <header className="flex justify-between items-center mb-8 border-b-2 border-slate-800 pb-6 bg-slate-900/40 backdrop-blur-md shadow-xl rounded-xl p-6">
        <div>
          <h1 className="text-3xl font-extrabold bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent flex items-center gap-3">
            <Server className="text-blue-500" />
            Zynaptrix Industrial Copilot
          </h1>
          <p className="text-slate-400 text-sm mt-1 font-medium tracking-wide">Generative AI Multi-Agent Orchestration Engine</p>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex bg-slate-800/50 rounded-xl p-1 border border-slate-700/50">
            {machines.map(m => (
              <button
                key={m.machine_id}
                onClick={() => dispatch(setCurrentMachineId(m.machine_id))}
                className={`px-4 py-2 rounded-lg text-xs font-bold transition-all ${currentMachineId === m.machine_id ? 'bg-blue-600 text-white shadow-lg' : 'text-slate-400 hover:text-slate-200'}`}
              >
                {m.name}
              </button>
            ))}
          </div>
          <div className={`px-5 py-2 rounded-full border shadow-lg flex items-center gap-2 font-bold tracking-widest ${
            systemState === 'NORMAL' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border-red-500/30 text-red-500 animate-pulse'
          }`}>
            {systemState === 'NORMAL' ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
            {systemState}
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-8 pb-10">
        
        {/* Left Column: Telemetry & Models */}
        <div className="xl:col-span-2 flex flex-col gap-8">
          
          {/* SENSOR KPI CARDS */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden group hover:border-blue-500/50 transition-colors">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                 <Thermometer size={64} className="text-blue-500" />
              </div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1 flex items-center gap-2"><Thermometer size={16} className="text-blue-400"/> Temperature</h3>
              <p className="text-4xl font-light text-slate-100 mt-2">{latestReading.temperature.toFixed(1)} <span className="text-lg text-slate-500">°C</span></p>
            </div>
            
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden group hover:border-emerald-500/50 transition-colors">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                 <Gauge size={64} className="text-emerald-500" />
              </div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1 flex items-center gap-2"><Gauge size={16} className="text-emerald-400"/> Motor Current</h3>
              <p className="text-4xl font-light text-slate-100 mt-2">{latestReading.current.toFixed(2)} <span className="text-lg text-slate-500">A</span></p>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden group hover:border-amber-500/50 transition-colors">
              <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                 <VibrateIcon size={64} className="text-amber-500" />
              </div>
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1 flex items-center gap-2"><VibrateIcon size={16} className="text-amber-400"/> Vibration</h3>
              <p className="text-4xl font-light text-slate-100 mt-2">{latestReading.vibration.toFixed(2)} <span className="text-lg text-slate-500">mm/s</span></p>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl -z-10 pointer-events-none translate-x-1/2 -translate-y-1/2"></div>
            <div className="flex justify-between items-center mb-6 z-10 relative">
              <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
                <Activity className="text-indigo-400" /> Live Sensor Telemetry Timeline
              </h2>
              <button
                onClick={toggleSimulation}
                className={`px-4 py-2 rounded-lg font-bold text-sm tracking-wide transition-all duration-300 flex items-center gap-2 shadow-lg ${isSimulating ? 'bg-red-500/20 text-red-500 border border-red-500/30 hover:bg-red-500/30' : 'bg-emerald-500/20 text-emerald-500 border border-emerald-500/30 hover:bg-emerald-500/30'}`}
              >
                {isSimulating ? <Square size={14} /> : <Play size={14} />}
                {isSimulating ? "Stop Simulator" : "Start Simulator"}
              </button>
            </div>
            <div className="h-[300px] w-full relative">
              {isMounted && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={filteredTelemetry} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#475569" tick={{fill: '#94a3b8'}} />
                    <YAxis stroke="#475569" tick={{fill: '#94a3b8'}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                    <Line type="monotone" dataKey="temperature" stroke="#3b82f6" strokeWidth={3} dot={false} name="Temp (°C)" />
                    <Line type="monotone" dataKey="current" stroke="#10b981" strokeWidth={3} dot={false} name="Current (A)" />
                    <Line type="monotone" dataKey="vibration" stroke="#f59e0b" strokeWidth={3} dot={false} name="Vibration (mm/s)" />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
               <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Anomaly Engine</h3>
               <div className="flex justify-between items-end">
                  <div>
                    <p className="text-4xl font-light text-slate-100">{anomalyScore.toFixed(3)}</p>
                    <p className="text-sm text-slate-500 mt-1">Reconstruction MSE</p>
                  </div>
                  <div className="h-12 w-12 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500 shadow-inner">
                    <ShieldCheck size={24} />
                  </div>
               </div>
            </div>

            {/* RAG DOCUMENT PIPELINE */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between relative">
               <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                 <UploadCloud size={16} className="text-indigo-400"/> RAG Knowledge Ingestion
               </h3>
               
               <div className="space-y-3 relative z-10">
                 <input 
                   type="text" 
                   value={manualId}
                   onChange={(e) => setManualId(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
                   placeholder="Manual ID (e.g. Pump_Manual_v1)" 
                   className="w-full bg-slate-950 border border-slate-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                   disabled={isUploading}
                 />
                 
                 <div className="flex gap-2 items-center">
                   <label className="flex-1 flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 cursor-pointer border border-slate-700 rounded-lg py-2 px-4 transition-colors text-sm font-semibold">
                     <FileText size={16} />
                     <span className="truncate max-w-[120px]">{uploadFile ? uploadFile.name : "Select PDF Document"}</span>
                     <input 
                       type="file" 
                       accept="application/pdf" 
                       className="hidden" 
                       onChange={(e) => setUploadFile(e.target.files ? e.target.files[0] : null)}
                       disabled={isUploading}
                     />
                   </label>
                   <button 
                     onClick={handleUpload}
                     disabled={!uploadFile || !manualId || isUploading}
                     className="bg-indigo-600 disabled:bg-slate-800 disabled:text-slate-500 hover:bg-indigo-500 text-white font-bold py-2 px-6 rounded-lg transition-colors flex items-center gap-2 shadow-lg shadow-indigo-500/20"
                   >
                     {isUploading ? <Loader2 size={16} className="animate-spin" /> : <UploadCloud size={16} />}
                     Upload
                   </button>
                 </div>
                 {uploadStatus && (
                   <p className={`text-xs font-semibold flex items-center gap-1 ${uploadStatus.includes('Successful') ? 'text-emerald-400' : uploadStatus.includes('Error') ? 'text-red-400' : 'text-blue-400'}`}>
                     {uploadStatus.includes('Successful') ? <CheckCircle size={14} /> : uploadStatus.includes('Error') ? <AlertTriangle size={14} /> : <Loader2 size={14} className="animate-spin"/>}
                     {uploadStatus}
                   </p>
                 )}
               </div>
            </div>
          </div>
        </div>

        {/* Right Column: AI Interaction */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden relative h-[calc(100vh-140px)]">
          <div className="absolute top-0 right-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
          <div className="p-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md z-10 flex justify-between items-center">
            <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
              <MessageSquare className="text-blue-400" /> Human-in-the-Loop
            </h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            {chatHistory.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-br-none' 
                    : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-bl-none leading-relaxed'}`}>
                  
                  {/* Human-readable interleaved content parser */}
                  <div className="text-sm space-y-4">
                    {msg.content.split(/(\[IMAGE[_\s-]?\d+\])/gi).map((part, partIdx) => {
                      const imageMatch = part.match(/\[IMAGE[_\s-]?(\d+)\]/i);
                      if (imageMatch && msg.images && msg.images[parseInt(imageMatch[1])]) {
                        const imgIdx = parseInt(imageMatch[1]);
                        return (
                          <div key={partIdx} className="my-4 rounded-xl overflow-hidden border border-slate-700 bg-slate-900/50 shadow-inner group">
                            <div className="bg-slate-800/80 px-3 py-1.5 text-[10px] font-bold text-slate-400 uppercase tracking-widest border-b border-slate-700 flex justify-between items-center text-xs">
                              <span>Technical Figure {imgIdx + 1}</span>
                              <span className="text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity">Visual Evidence</span>
                            </div>
                            <img 
                              src={msg.images[imgIdx]} 
                              alt={`Diagnostic Figure ${imgIdx + 1}`}
                              className="w-full h-auto max-h-[400px] object-contain cursor-zoom-in hover:scale-[1.01] transition-transform duration-300"
                              onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                            />
                          </div>
                        );
                      }
                      return (
                        <div key={partIdx} className="part-content">
                          {part.split('\n').map((line, lineIdx) => {
                            if (line.trim().startsWith('###')) {
                              return <h3 key={lineIdx} className="text-lg font-bold text-blue-400 mt-4 border-b border-slate-700/50 pb-1">{line.replace('###', '').trim()}</h3>;
                            }
                            if (line.trim().startsWith('##')) {
                              return <h2 key={lineIdx} className="text-xl font-black text-slate-100 mt-6 flex items-center gap-2">
                                <Activity size={18} className="text-indigo-500" /> {line.replace('##', '').trim()}
                              </h2>;
                            }
                            if (line.includes('**')) {
                              const segments = line.split(/(\*\*.*?\*\*)/g);
                              return (
                                <p key={lineIdx} className="whitespace-pre-wrap">
                                  {segments.map((seg, segIdx) => {
                                    if (seg.startsWith('**') && seg.endsWith('**')) {
                                      return <strong key={segIdx} className="text-blue-300 font-bold">{seg.slice(2, -2)}</strong>;
                                    }
                                    return seg;
                                  })}
                                </p>
                              );
                            }
                            return <p key={lineIdx} className="whitespace-pre-wrap">{line}</p>;
                          })}
                        </div>
                      );
                    })}
                  </div>

                  {/* Supplemental Images */}
                  {msg.images && msg.images.length > 0 && !msg.content.match(/\[IMAGE[_\s-]?\d+\]/i) && (
                    <div className="mt-6 pt-4 border-t border-slate-700/50 grid grid-cols-1 gap-4">
                      <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Supplemental Records</p>
                      {msg.images.map((imgUrl, imgIdx) => (
                        <div key={imgIdx} className="rounded-xl overflow-hidden border border-slate-700 bg-slate-900/50">
                          <img 
                            src={imgUrl} 
                            alt={`Retrieved Diagram ${imgIdx + 1}`}
                            className="w-full h-auto max-h-64 object-contain"
                          />
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          <div className="p-4 bg-slate-900 border-t border-slate-800 z-10">
            <div className="flex items-center gap-3">
              <input
                type="text"
                placeholder="Ask the Diagnostic Agent..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-5 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 transition-all text-slate-200"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleManualInquiry()}
              />
              <button 
                onClick={handleManualInquiry}
                className="bg-blue-600 hover:bg-blue-500 text-white p-3 rounded-xl transition-colors shadow-lg shadow-blue-500/20"
              >
                <Send size={18} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

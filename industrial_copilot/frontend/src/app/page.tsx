"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, ShieldCheck, Server, MessageSquare, Send, Thermometer, Gauge, Activity as VibrateIcon, UploadCloud, FileText, CheckCircle, Loader2, Play, Square } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { addChatMessage, addTelemetry, setSystemState, setAnomalyScore } from '../store/slices/copilotSlice';

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  const { telemetry, chatHistory, systemState, anomalyScore, activeAgents } = useSelector((state: RootState) => state.copilot);
  
  const [query, setQuery] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Upload States
  const [manualId, setManualId] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  
  const [isSimulating, setIsSimulating] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  // WebSocket Telemetry Stream
  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: NodeJS.Timeout;

    const connectWS = () => {
      const wsUrl = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8500').replace('http', 'ws');
      ws = new WebSocket(`${wsUrl}/ws/telemetry`);
      
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          
          if (parsed.type === "telemetry") {
            const data = parsed.data;
            dispatch(addTelemetry({
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
              temperature: data.temperature,
              pressure: data.pressure,
              vibration: data.vibration
            }));
          } else if (parsed.type === "anomaly_alert") {
            const result = parsed.data;
            dispatch(setSystemState('ANOMALY'));
            dispatch(setAnomalyScore(result.anomaly_score || 0.99));
            
            const alertMessage = `🚨 **CRITICAL ANOMALY EVENT TRIGGERED**\n\n**Suspected System**: ${result.suspect_sensor}\n\n**AI Diagnostic Procedure:**\n${result.strategy || result.final_execution_plan}\n\n**RAG Knowledge Base Context:**\n${result.rag_advice || "No context found."}`;
            dispatch(addChatMessage({ 
              role: 'agent', 
              content: alertMessage,
              images: result.retrieved_images || []
            }));
          }
        } catch (err) {
          console.error("Telemetry parsing error:", err);
        }
      };

      ws.onclose = () => {
        reconnectTimer = setTimeout(connectWS, 3000); // Self-heal after backend crash/restart
      };
      
      ws.onerror = () => {
        ws.close();
      };
    };

    connectWS();

    return () => {
      clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null; // Prevent infinite re-mounting loops
        ws.close();
      }
    };
  }, [dispatch]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const latestReading = telemetry[telemetry.length - 1] || { temperature: 0, pressure: 0, vibration: 0 };

  const handleManualInquiry = async () => {
    if (!query) return;
    dispatch(addChatMessage({ role: 'user', content: query }));
    setQuery('');
    
    // For simplicity, we just push the analyzing message...
    dispatch(addChatMessage({ role: 'agent', content: 'Analyzing factory status deeply using Multi-Agent LangGraph...' }));
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8500';
      const res = await fetch(`${apiUrl}/api/copilot/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          machine_state: systemState === "NORMAL" ? "manual_inquiry_normal" : "anomaly_investigation",
          anomaly_score: anomalyScore,
          suspect_sensor: "User Query",
          recent_readings: latestReading
        })
      });
      const data = await res.json();
      dispatch(addChatMessage({ 
        role: 'agent', 
        content: data.graph_result?.final_execution_plan || "Diagnostics complete.",
        images: data.graph_result?.retrieved_images || []
      }));
    } catch (e) {
      dispatch(addChatMessage({ role: 'agent', content: 'Connection to AI Orchestrator Failed.' }));
    }
  };

  const handleUpload = async () => {
    if (!uploadFile || !manualId) return;
    setIsUploading(true);
    setUploadStatus("Uploading to YOLOv8 Pipeline...");

    const formData = new FormData();
    formData.append("manual_id", manualId);
    formData.append("file", uploadFile);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8500';
      const res = await fetch(`${apiUrl}/ingest-manual`, {
         method: 'POST',
         body: formData
      });
      
      if (res.ok) {
         setUploadStatus("Ingestion Successful!");
         dispatch(addChatMessage({ role: 'agent', content: `📚 Manual "${manualId}" successfully ingested and vectorized.` }));
         setTimeout(() => { setUploadStatus(null); setManualId(""); setUploadFile(null); }, 3000);
      } else {
         const errorData = await res.json();
         setUploadStatus(`Error: ${errorData.detail || "Upload failed"}`);
      }
    } catch(e) {
      setUploadStatus("Connection Error to Backend.");
    }
    setIsUploading(false);
  };

  const toggleSimulation = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8500';
      const endpoint = isSimulating ? '/api/simulator/stop' : '/api/simulator/start';
      const res = await fetch(`${apiUrl}${endpoint}`, { method: 'POST' });
      if (res.ok) {
        setIsSimulating(!isSimulating);
      }
    } catch (e) {
      console.error("Failed to toggle simulation", e);
    }
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
        <div className={`px-5 py-2 rounded-full border shadow-lg flex items-center gap-2 font-bold tracking-widest ${
          systemState === 'NORMAL' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-red-500/10 border-red-500/30 text-red-500 animate-pulse'
        }`}>
          {systemState === 'NORMAL' ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
          {systemState}
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
              <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-1 flex items-center gap-2"><Gauge size={16} className="text-emerald-400"/> Pressure</h3>
              <p className="text-4xl font-light text-slate-100 mt-2">{latestReading.pressure.toFixed(1)} <span className="text-lg text-slate-500">bar</span></p>
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
                  <LineChart data={telemetry} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#475569" tick={{fill: '#94a3b8'}} />
                    <YAxis stroke="#475569" tick={{fill: '#94a3b8'}} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                      itemStyle={{ color: '#e2e8f0' }}
                    />
                    <Line type="monotone" dataKey="temperature" stroke="#3b82f6" strokeWidth={3} dot={false} name="Temp (°C)" />
                    <Line type="monotone" dataKey="pressure" stroke="#10b981" strokeWidth={3} dot={false} name="Pressure (bar)" />
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
                   <p className={`text-xs font-semibold flex items-center gap-1 ${uploadStatus.includes('Success') ? 'text-emerald-400' : uploadStatus.includes('Error') ? 'text-red-400' : 'text-blue-400'}`}>
                     {uploadStatus.includes('Success') ? <CheckCircle size={14} /> : uploadStatus.includes('Error') ? <AlertTriangle size={14} /> : <Loader2 size={14} className="animate-spin"/>}
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
                    : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-bl-none leading-relaxed'                }`}>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  
                  {msg.images && msg.images.length > 0 && (
                    <div className="mt-4 grid grid-cols-1 gap-3">
                      {msg.images.map((imgUrl, imgIdx) => (
                        <div key={imgIdx} className="rounded-lg overflow-hidden border border-slate-700 bg-slate-900/50">
                          <img 
                            src={imgUrl} 
                            alt={`Retrieved Diagram ${imgIdx + 1}`}
                            className="w-full h-auto max-h-64 object-contain cursor-zoom-in hover:scale-[1.02] transition-transform"
                            onError={(e) => {
                              (e.target as HTMLImageElement).style.display = 'none';
                            }}
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

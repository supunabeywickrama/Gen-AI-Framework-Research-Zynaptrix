"use client"
import React, { useState, useEffect, useRef } from 'react';
import { Activity, AlertTriangle, ShieldCheck, Zap, Server, MessageSquare, Send } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../store/store';
import { addChatMessage } from '../store/slices/copilotSlice';

export default function IndustrialCopilotDashboard() {
  const dispatch = useDispatch<AppDispatch>();
  const { telemetry, chatHistory, systemState, anomalyScore, activeAgents } = useSelector((state: RootState) => state.copilot);
  
  const [query, setQuery] = useState('');
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleManualInquiry = async () => {
    if (!query) return;
    dispatch(addChatMessage({ role: 'user', content: query }));
    setQuery('');
    
    // Call LangGraph Backend
    // For simplicity, we just push the analyzing message...
    dispatch(addChatMessage({ role: 'agent', content: 'Analyzing factory status deeply using Multi-Agent LangGraph...' }));
    
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const res = await fetch(`${apiUrl}/api/copilot/invoke`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          machine_state: systemState === "NORMAL" ? "manual_inquiry_normal" : "anomaly_investigation",
          anomaly_score: anomalyScore,
          suspect_sensor: "User Query",
          recent_readings: telemetry[telemetry.length - 1]
        })
      });
      const data = await res.json();
      
      // Dispatch final result natively
      dispatch(addChatMessage({ role: 'agent', content: data.graph_result?.final_execution_plan || "Diagnostics complete." }));
    } catch (e) {
      dispatch(addChatMessage({ role: 'agent', content: 'Connection to AI Orchestrator Failed.' }));
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans selection:bg-blue-500/30">
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

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-[calc(100vh-140px)]">
        
        {/* Left Column: Telemetry & Models */}
        <div className="lg:col-span-2 flex flex-col gap-8">
          
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/5 rounded-full blur-3xl -z-10 absolute pointer-events-none translate-x-1/2 -translate-y-1/2"></div>
            <h2 className="text-xl font-bold mb-6 flex items-center gap-2 text-slate-100">
              <Activity className="text-indigo-400" /> Live Sensor Telemetry
            </h2>
            <div className="h-72 w-full">
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
                  <Line type="monotone" dataKey="pressure" stroke="#10b981" strokeWidth={3} dot={false} name="Pressure (psi)" />
                  <Line type="monotone" dataKey="vibration" stroke="#f59e0b" strokeWidth={3} dot={false} name="Vibration (mm/s)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
               <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Anomaly Engine</h3>
               <div className="flex justify-between items-end">
                  <div>
                    <p className="text-4xl font-light text-slate-100">{anomalyScore.toFixed(3)}</p>
                    <p className="text-sm text-slate-500 mt-1">Reconstruction MSE</p>
                  </div>
                  <div className="h-12 w-12 rounded-full bg-emerald-500/20 flex items-center justify-center text-emerald-500">
                    <ShieldCheck size={24} />
                  </div>
               </div>
            </div>
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl">
               <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-widest mb-4">Active AI Agents</h3>
               <div className="flex gap-2">
                 {activeAgents.map(agent => (
                   <span key={agent} className="px-2 py-1 text-xs font-bold rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                     {agent}
                   </span>
                 ))}
               </div>
            </div>
          </div>
        </div>

        {/* Right Column: AI Interaction */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden relative">
          <div className="absolute top-0 right-0 w-full h-1 bg-gradient-to-r from-blue-500 to-indigo-500"></div>
          <div className="p-6 border-b border-slate-800 bg-slate-900/50 backdrop-blur-md z-10">
            <h2 className="text-xl font-bold flex items-center gap-2 text-slate-100">
              <MessageSquare className="text-blue-400" /> Human-in-the-Loop
            </h2>
          </div>
          
          <div className="flex-1 overflow-y-auto p-6 space-y-6 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]">
            {chatHistory.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${
                  msg.role === 'user' 
                    ? 'bg-blue-600 text-white rounded-br-none' 
                    : 'bg-slate-800 text-slate-200 border border-slate-700/50 rounded-bl-none leading-relaxed'                }`}>
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
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

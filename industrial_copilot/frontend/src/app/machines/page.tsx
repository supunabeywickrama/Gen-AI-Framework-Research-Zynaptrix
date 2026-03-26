"use client"
import React, { useState, useEffect } from 'react';
import { Cog, Plus, ShieldCheck, MapPin, FileCode, CheckCircle, AlertTriangle, Loader2, Factory, Trash2, Edit, X } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState, AppDispatch } from '../../store/store';
import { fetchMachines, registerMachine, deleteMachine, Machine } from '../../store/slices/machineSlice';

export default function MachineRegistryPage() {
  const dispatch = useDispatch<AppDispatch>();
  const { machines, loading, error } = useSelector((state: RootState) => state.machines);
  
  const [formData, setFormData] = useState<Machine>({
    machine_id: '',
    name: '',
    location: '',
    manual_id: ''
  });

  const [registerStatus, setRegisterStatus] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => {
    dispatch(fetchMachines());
  }, [dispatch]);

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault();
    setRegisterStatus(isEditing ? 'Updating...' : 'Registering...');
    try {
      await dispatch(registerMachine(formData)).unwrap();
      setRegisterStatus(isEditing ? 'Success! Machine updated.' : 'Success! Machine registered.');
      setFormData({ machine_id: '', name: '', location: '', manual_id: '' });
      setIsEditing(false);
      setTimeout(() => setRegisterStatus(null), 3000);
    } catch (err) {
      setRegisterStatus(`Error: ${err}`);
      setTimeout(() => setRegisterStatus(null), 5000);
    }
  };

  const startEdit = (machine: Machine) => {
    setFormData(machine);
    setIsEditing(true);
    setRegisterStatus(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const cancelEdit = () => {
    setFormData({ machine_id: '', name: '', location: '', manual_id: '' });
    setIsEditing(false);
  };

  const handleDelete = async (id: string) => {
    if (!confirm(`Are you sure you want to decommission asset ${id}? This cannot be undone.`)) return;
    setDeletingId(id);
    try {
      await dispatch(deleteMachine(id)).unwrap();
    } catch (err) {
      alert(`Failed to delete: ${err}`);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <header className="mb-12">
        <h1 className="text-4xl font-black text-white mb-2 flex items-center gap-3">
          <Factory className="text-blue-500" size={36} />
          Asset Registry
        </h1>
        <p className="text-slate-400 text-lg font-medium">Manage and enroll physical assets into the Industrial Copilot ecosystem.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Registration Form */}
        <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl h-fit sticky top-24">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              {isEditing ? <Edit size={20} className="text-amber-400" /> : <Plus size={20} className="text-blue-400" />}
              {isEditing ? 'Update Machine' : 'Register New Machine'}
            </h2>
            {isEditing && (
              <button 
                onClick={cancelEdit}
                className="text-slate-500 hover:text-white transition-colors"
                title="Cancel Edit"
              >
                <X size={20} />
              </button>
            )}
          </div>

          <form onSubmit={handleRegister} className="space-y-5">
            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Machine ID</label>
              <input 
                type="text" 
                required
                disabled={isEditing}
                value={formData.machine_id}
                onChange={(e) => setFormData({...formData, machine_id: e.target.value.toUpperCase().replace(/\s/g, '_')})}
                placeholder="e.g. DRILL-004" 
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-500 transition-all text-white disabled:opacity-50 disabled:bg-slate-900 cursor-not-allowed"
              />
              {isEditing && <p className="text-[10px] text-amber-500/70 mt-1 font-medium italic">Asset ID cannot be modified after initial enrollment.</p>}
            </div>
            {/* ... other fields remain same ... */}

            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Machine Name</label>
              <input 
                type="text" 
                required
                value={formData.name}
                onChange={(e) => setFormData({...formData, name: e.target.value})}
                placeholder="e.g. Precision Lathe" 
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-blue-500 transition-all text-white"
              />
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Location</label>
              <div className="relative">
                <MapPin className="absolute left-4 top-3.5 text-slate-500" size={16} />
                <input 
                  type="text" 
                  required
                  value={formData.location}
                  onChange={(e) => setFormData({...formData, location: e.target.value})}
                  placeholder="e.g. Factory Floor B" 
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-12 pr-4 py-3 text-sm focus:outline-none focus:border-blue-500 transition-all text-white"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Target Manual ID</label>
              <div className="relative">
                <FileCode className="absolute left-4 top-3.5 text-slate-500" size={16} />
                <input 
                  type="text" 
                  required
                  value={formData.manual_id}
                  onChange={(e) => setFormData({...formData, manual_id: e.target.value})}
                  placeholder="e.g. Lathe_Manual_V1" 
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-12 pr-4 py-3 text-sm focus:outline-none focus:border-blue-500 transition-all text-white"
                />
              </div>
            </div>

            <button 
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 text-white font-bold py-4 px-6 rounded-2xl transition-all flex items-center justify-center gap-3 shadow-xl shadow-blue-500/20"
            >
              {loading ? <Loader2 size={20} className="animate-spin" /> : <Plus size={20} />}
              Enroll Machine
            </button>

            {registerStatus && (
              <div className={`p-4 rounded-xl text-sm font-bold flex items-center gap-2 ${registerStatus.includes('Error') ? 'bg-red-500/10 text-red-400 border border-red-500/20' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'}`}>
                {registerStatus.includes('Error') ? <AlertTriangle size={16} /> : <CheckCircle size={16} />}
                {registerStatus}
              </div>
            )}
          </form>
        </div>

        {/* Live Fleet Table */}
        <div className="lg:col-span-2">
          <div className="bg-slate-900 border border-slate-800 rounded-3xl overflow-hidden shadow-2xl">
            <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-slate-900/50">
               <h2 className="text-xl font-bold text-white flex items-center gap-2">
                 <ShieldCheck size={20} className="text-emerald-400" />
                 Live Fleet Registry
               </h2>
               <span className="bg-slate-800 text-slate-400 text-[10px] px-3 py-1 rounded-full font-black uppercase tracking-tighter">
                 {machines.length} Total Assets
               </span>
            </div>
            
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse min-w-[700px]">
                <thead>
                  <tr className="bg-slate-950/50 border-b border-slate-800/50">
                    <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Asset ID</th>
                    <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Asset Name & Manual</th>
                    <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em]">Deployment Location</th>
                    <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] text-center">Status</th>
                    <th className="px-6 py-5 text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/30">
                  {machines.map((machine) => (
                    <tr key={machine.machine_id} className="hover:bg-blue-500/[0.02] transition-colors group">
                      <td className="px-6 py-6 vertical-top">
                        <span className="font-mono text-[11px] text-blue-400 font-bold bg-blue-400/5 px-2.5 py-1.5 rounded-lg border border-blue-400/20 whitespace-nowrap leading-none shadow-sm shadow-blue-900/10">
                          {machine.machine_id}
                        </span>
                      </td>
                      <td className="px-6 py-6">
                        <div className="text-white font-bold text-sm tracking-tight">{machine.name}</div>
                        <div className="text-[10px] text-slate-500 font-mono mt-1.5 opacity-70 flex items-center gap-1.5">
                          <FileCode size={10} />
                          {machine.manual_id}
                        </div>
                      </td>
                      <td className="px-6 py-6">
                        <div className="flex items-center gap-2 text-slate-400 text-xs font-medium bg-slate-800/20 w-fit px-3 py-1.5 rounded-full border border-slate-800/30">
                          <MapPin size={12} className="text-slate-500" />
                          {machine.location}
                        </div>
                      </td>
                      <td className="px-6 py-6 text-center">
                        <div className="inline-flex items-center gap-2.5 text-emerald-400 text-[10px] font-black bg-emerald-400/5 px-3.5 py-2 rounded-xl border border-emerald-400/20 uppercase tracking-tighter">
                          <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse"></div>
                          Live
                        </div>
                      </td>
                      <td className="px-6 py-6 text-right flex items-center justify-end gap-2">
                        <button 
                          onClick={() => startEdit(machine)}
                          className="p-2.5 text-slate-500 hover:text-amber-400 hover:bg-amber-400/10 rounded-xl transition-all border border-transparent hover:border-amber-500/20"
                          title="Edit Machine Details"
                        >
                          <Edit size={18} />
                        </button>
                        <button 
                          onClick={() => handleDelete(machine.machine_id)}
                          disabled={deletingId === machine.machine_id}
                          className="p-2.5 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded-xl transition-all disabled:opacity-50 border border-transparent hover:border-red-500/20"
                          title="Decommission Asset"
                        >
                          {deletingId === machine.machine_id ? <Loader2 size={18} className="animate-spin" /> : <Trash2 size={18} />}
                        </button>
                      </td>
                    </tr>
                  ))}
                  {machines.length === 0 && (
                    <tr>
                      <td colSpan={4} className="px-6 py-12 text-center text-slate-500 italic">
                        No assets discovered in the network registry.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="mt-6 flex gap-4 p-6 bg-blue-600/5 border border-blue-500/20 rounded-3xl">
            <div className="h-10 w-10 rounded-2xl bg-blue-600/20 flex items-center justify-center text-blue-400 shrink-0">
               <ShieldCheck size={24} />
            </div>
            <div>
              <p className="text-white font-bold text-sm">Automated Neural Handshake</p>
              <p className="text-slate-500 text-xs mt-1 leading-relaxed">
                Registered machines are automatically assigned a zero-centered normalization profile. Neural Autoencoders will begin reconstruction scoring once the first telemetry frame is received.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

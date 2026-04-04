"use client";

import React from 'react';

const GalaxyBackground = () => {
  return (
    <div className="fixed inset-0 pointer-events-none select-none overflow-hidden" 
         style={{ zIndex: -1, background: 'radial-gradient(ellipse at bottom, #0d1d31 0%, #000c1d 100%)' }}>
      
      {/* 🌌 Deep Nebula Layer 1 */}
      <div className="absolute inset-0 opacity-40 mix-blend-screen animate-pulse" 
           style={{ 
             background: 'radial-gradient(circle at 15% 15%, #1e40af 0%, transparent 60%), radial-gradient(circle at 85% 85%, #4338ca 0%, transparent 60%), radial-gradient(circle at 50% 50%, #1e1b4b 0%, transparent 70%)',
             filter: 'blur(60px)'
           }} />
      
      {/* 🌌 Deep Nebula Layer 2 (Indigo/Violet) */}
      <div className="absolute inset-0 opacity-30 mix-blend-overlay animate-slow-spin-slow" 
           style={{ 
             background: 'radial-gradient(circle at 70% 30%, #312e81 0%, transparent 50%), radial-gradient(circle at 30% 70%, #1e1b4b 0%, transparent 50%)',
             filter: 'blur(40px)'
           }} />

      {/* ⭐ Star Layer 1: Tiny distant stars (Parallax Slow) */}
      <div className="absolute inset-0 opacity-80" 
           style={{ 
             backgroundImage: 'radial-gradient(1px 1px at 20px 30px, #fff, rgba(0,0,0,0)), radial-gradient(1px 1px at 150px 100px, #fff, rgba(0,0,0,0)), radial-gradient(1px 1px at 300px 450px, #fff, rgba(0,0,0,0)), radial-gradient(1px 1px at 500px 300px, #fff, rgba(0,0,0,0)), radial-gradient(1px 1px at 700px 150px, #fff, rgba(0,0,0,0))',
             backgroundSize: '300px 300px',
             backgroundRepeat: 'repeat',
             animation: 'move-stars 150s linear infinite'
           }} />

      {/* ⭐ Star Layer 2: Medium stars (Parallax Medium) */}
      <div className="absolute inset-0 opacity-60" 
           style={{ 
             backgroundImage: 'radial-gradient(1.5px 1.5px at 100px 50px, #fff, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 250px 200px, #fff, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 450px 350px, #fff, rgba(0,0,0,0)), radial-gradient(1.5px 1.5px at 600px 50px, #fff, rgba(0,0,0,0))',
             backgroundSize: '450px 450px',
             backgroundRepeat: 'repeat',
             animation: 'move-stars 100s linear infinite'
           }} />

      {/* ⭐ Star Layer 3: Occasional Glowy Stars */}
      <div className="absolute inset-0 opacity-40" 
           style={{ 
             backgroundImage: 'radial-gradient(2px 2px at 80px 160px, #fff, rgba(0,0,0,0)), radial-gradient(2px 2px at 400px 400px, #fff, rgba(0,0,0,0)), radial-gradient(2.5px 2.5px at 800px 600px, #fff, rgba(0,0,0,0))',
             backgroundSize: '800px 800px',
             backgroundRepeat: 'repeat'
           }} />

      {/* 🌫️ Stardust Fog Pattern */}
      <div className="absolute inset-0 opacity-[0.03] mix-blend-lighten pointer-events-none" 
           style={{ background: 'url("https://www.transparenttextures.com/patterns/stardust.png") repeat' }} />

      <style jsx>{`
        @keyframes move-stars {
          from { transform: translateY(0); }
          to { transform: translateY(-1000px); }
        }
        @keyframes slow-spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .animate-slow-spin-slow {
          animation: slow-spin-slow 200s linear infinite;
        }
      `}</style>
    </div>
  );
};

export default GalaxyBackground;

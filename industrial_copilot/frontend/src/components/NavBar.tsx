"use client"
import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { LayoutDashboard, Database, Cog, Factory } from 'lucide-react';

const NavBar = () => {
  const pathname = usePathname();

  const navItems = [
    { name: 'Dashboard', href: '/', icon: LayoutDashboard },
    { name: 'Machine Management', href: '/machines', icon: Cog },
    { name: 'Knowledge Ingestion', href: '/ingestion', icon: Database },
  ];

  return (
    <nav className="bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-8 py-4 sticky top-0 z-50 flex justify-between items-center shadow-2xl">
      <div className="flex items-center gap-3">
        <div className="bg-blue-600 p-2 rounded-lg shadow-lg shadow-blue-500/20">
          <Factory className="text-white" size={20} />
        </div>
        <span className="text-xl font-black bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent tracking-tight">
          ZYNAPTRIX <span className="text-blue-500 font-light">COPILOT</span>
        </span>
      </div>

      <div className="flex items-center gap-2">
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          
          return (
            <Link
              key={item.name}
              href={item.href}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300 text-sm font-bold tracking-wide ${
                isActive 
                  ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' 
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800'
              }`}
            >
              <Icon size={18} />
              {item.name}
            </Link>
          );
        })}
      </div>

      <div className="flex items-center gap-4">
        <div className="h-4 w-[1px] bg-slate-800 mx-2"></div>
        <button className="text-slate-400 hover:text-white transition-colors">
          <Cog size={20} />
        </button>
      </div>
    </nav>
  );
};

export default NavBar;

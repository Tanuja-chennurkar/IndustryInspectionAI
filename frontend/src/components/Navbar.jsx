import React from 'react';
import { LayoutDashboard, Scan, History, BarChart3, ShieldCheck } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, isBackendConnected }) {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'inspect', label: 'Inspect', icon: Scan },
    { id: 'history', label: 'History', icon: History },
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  ];

  return (
    <header className="navbar">
      <div className="nav-brand">
        <ShieldCheck className="w-7 h-7 text-blue-500" />
        <span>IndustryInspection<span style={{ color: '#3b82f6' }}>AI</span></span>
        <span className="nav-brand-badge">VisA Self-Supervised</span>
      </div>

      <nav className="nav-tabs">
        {tabs.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`nav-tab ${activeTab === t.id ? 'active' : ''}`}
            >
              <Icon size={18} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </nav>

      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
        <span
          style={{
            width: '10px',
            height: '10px',
            borderRadius: '50%',
            backgroundColor: isBackendConnected ? '#10b981' : '#ef4444',
            display: 'inline-block'
          }}
        ></span>
        <span style={{ color: isBackendConnected ? '#34d399' : '#f87171' }}>
          {isBackendConnected ? 'Backend Active (FastAPI)' : 'Connecting to API...'}
        </span>
      </div>
    </header>
  );
}

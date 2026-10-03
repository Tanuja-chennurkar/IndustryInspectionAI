import React from 'react';

export default function KpiCard({ title, value, icon: Icon, color, subtext }) {
  return (
    <div className="kpi-card">
      <div>
        <div className="kpi-title">{title}</div>
        <div className="kpi-value" style={{ color: color || '#ffffff' }}>
          {value}
        </div>
        {subtext && <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.25rem' }}>{subtext}</div>}
      </div>
      <div
        className="kpi-icon-pill"
        style={{
          backgroundColor: color ? `${color}20` : '#334155',
          color: color || '#94a3b8'
        }}
      >
        <Icon size={24} />
      </div>
    </div>
  );
}

import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts';
import { fetchStatistics, fetchInspections } from '../services/api';

const COLORS = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4'];

export default function AnalyticsPage() {
  const [stats, setStats] = useState(null);
  const [inspections, setInspections] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadAnalytics() {
      try {
        const [s, h] = await Promise.all([
          fetchStatistics(),
          fetchInspections('', '', 500)
        ]);
        setStats(s);
        setInspections(h);
      } catch (err) {
        console.error('Analytics load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadAnalytics();
  }, []);

  if (loading) {
    return <div className="page-content" style={{ textAlign: 'center', padding: '4rem' }}>Loading Telemetry Analytics...</div>;
  }

  // 1. Anomaly Rate by Category
  const categoryData = Object.entries(stats?.category_breakdown || {}).map(([cat, val]) => ({
    name: cat.toUpperCase(),
    Total: val.total,
    Normal: val.normal,
    Anomaly: val.anomaly,
    Rate: val.total > 0 ? Number(((val.anomaly / val.total) * 100).toFixed(1)) : 0
  }));

  // 2. Anomaly Score Distribution Bins
  const scoreBins = [
    { range: '0.0 - 0.2', count: 0 },
    { range: '0.2 - 0.4', count: 0 },
    { range: '0.4 - 0.6', count: 0 },
    { range: '0.6 - 0.8', count: 0 },
    { range: '0.8 - 1.0', count: 0 },
  ];
  inspections.forEach((item) => {
    const s = item.anomaly_score || 0;
    if (s < 0.2) scoreBins[0].count += 1;
    else if (s < 0.4) scoreBins[1].count += 1;
    else if (s < 0.6) scoreBins[2].count += 1;
    else if (s < 0.8) scoreBins[3].count += 1;
    else scoreBins[4].count += 1;
  });

  // 3. Status Breakdown Pie Chart
  const pieData = [
    { name: 'NORMAL', value: stats?.normal_count || 0 },
    { name: 'ANOMALY', value: stats?.anomaly_count || 0 }
  ];

  return (
    <div className="page-content">
      <div className="page-title-section">
        <h1 className="page-title">Industrial Quality Analytics & Telemetry</h1>
        <p className="page-subtitle">Recharts telemetry visualization across all 12 VisA product categories</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
        {/* Chart 1: Anomaly Rate per Category */}
        <div className="panel-card">
          <div className="panel-header">
            <span>Anomalies by Category</span>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
              <BarChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
                <Bar dataKey="Normal" fill="#10b981" />
                <Bar dataKey="Anomaly" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Anomaly Score Distribution */}
        <div className="panel-card">
          <div className="panel-header">
            <span>Anomaly Score Distribution</span>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
              <BarChart data={scoreBins}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="range" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        {/* Chart 3: Normal vs Anomaly Proportion Pie */}
        <div className="panel-card">
          <div className="panel-header">
            <span>Product Quality Proportion</span>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                  <Cell fill="#10b981" />
                  <Cell fill="#ef4444" />
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Anomaly Rate Percentage per Category */}
        <div className="panel-card">
          <div className="panel-header">
            <span>Defect Rate Percentage (%) by Category</span>
          </div>
          <div style={{ height: '300px', width: '100%' }}>
            <ResponsiveContainer>
              <LineChart data={categoryData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" unit="%" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                <Line type="monotone" dataKey="Rate" stroke="#f59e0b" strokeWidth={3} dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

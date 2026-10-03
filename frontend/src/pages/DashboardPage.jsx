import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle, AlertTriangle, Percent, Cpu, CpuIcon } from 'lucide-react';
import KpiCard from '../components/KpiCard';
import { fetchStatistics, fetchDefects, fetchModels } from '../services/api';

export default function DashboardPage({ onNavigateToInspect }) {
  const [stats, setStats] = useState(null);
  const [defects, setDefects] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        const [s, d, m] = await Promise.all([
          fetchStatistics(),
          fetchDefects(5),
          fetchModels()
        ]);
        setStats(s);
        setDefects(d);
        setModels(m);
      } catch (err) {
        console.error('Dashboard data load error:', err);
      } finally {
        setLoading(false);
      }
    }
    loadDashboardData();
  }, []);

  if (loading) {
    return <div className="page-content" style={{ textAlign: 'center', padding: '4rem' }}>Loading Industrial Overview...</div>;
  }

  return (
    <div className="page-content">
      <div className="page-title-section" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1 className="page-title">Executive Inspection Overview</h1>
          <p className="page-subtitle">Real-time VisA Dataset Self-Supervised Anomaly Detection Telemetry</p>
        </div>
        <button
          onClick={onNavigateToInspect}
          style={{
            backgroundColor: '#3b82f6',
            color: '#fff',
            border: 'none',
            padding: '0.65rem 1.25rem',
            borderRadius: '8px',
            fontWeight: 600,
            cursor: 'pointer'
          }}
        >
          + Run New Inspection
        </button>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <KpiCard
          title="TOTAL INSPECTIONS"
          value={stats?.total_inspections ?? 0}
          icon={Activity}
          color="#3b82f6"
          subtext="Across all 12 object categories"
        />
        <KpiCard
          title="NORMAL PRODUCTS"
          value={stats?.normal_count ?? 0}
          icon={CheckCircle}
          color="#10b981"
          subtext="Passed quality control"
        />
        <KpiCard
          title="ANOMALIES DETECTED"
          value={stats?.anomaly_count ?? 0}
          icon={AlertTriangle}
          color="#ef4444"
          subtext="Flagged for review"
        />
        <KpiCard
          title="DEFECT RATE"
          value={`${stats?.defect_rate_percentage ?? 0}%`}
          icon={Percent}
          color="#f59e0b"
          subtext="Anomaly ratio"
        />
        <KpiCard
          title="AVG ANOMALY SCORE"
          value={stats?.average_anomaly_score ?? 0}
          icon={Cpu}
          color="#a855f7"
          subtext="SSIM + L1 error metric"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '2rem' }}>
        {/* Recent Defects Table */}
        <div className="panel-card">
          <div className="panel-header">
            <span>Recent Anomalous Inspections</span>
            <span style={{ fontSize: '0.8rem', color: '#f87171' }}>{defects.length} Flagged Items</span>
          </div>
          {defects.length === 0 ? (
            <div style={{ color: '#94a3b8', fontSize: '0.9rem', padding: '1rem 0' }}>
              No defects detected yet. Run an inspection to populate defect telemetry.
            </div>
          ) : (
            <table className="custom-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Category</th>
                  <th>Score</th>
                  <th>Defect Area</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {defects.map((d) => (
                  <tr key={d.inspection_id}>
                    <td>{d.inspection_id}</td>
                    <td style={{ textTransform: 'capitalize' }}>{d.category}</td>
                    <td>{d.anomaly_score}</td>
                    <td>{d.defect_area_percentage}%</td>
                    <td>
                      <span className="badge-status badge-anomaly">ANOMALY</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* VisA Models Registry Table */}
        <div className="panel-card">
          <div className="panel-header">
            <span>VisA 12 Category Models Registry</span>
            <span style={{ fontSize: '0.8rem', color: '#34d399' }}>12 Active Keras Models</span>
          </div>
          <table className="custom-table">
            <thead>
              <tr>
                <th>Category</th>
                <th>Version</th>
                <th>Threshold</th>
                <th>Input Shape</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.category}>
                  <td style={{ fontWeight: 600, textTransform: 'capitalize' }}>{m.category}</td>
                  <td>{m.version}</td>
                  <td>{m.threshold}</td>
                  <td>{m.input_shape.join('x')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

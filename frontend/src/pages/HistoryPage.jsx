import React, { useEffect, useState } from 'react';
import { Filter, Eye, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import { fetchInspections } from '../services/api';

const VISA_CATEGORIES = [
  'all',
  'candle',
  'capsules',
  'cashew',
  'chewinggum',
  'fryum',
  'macaroni1',
  'macaroni2',
  'pcb1',
  'pcb2',
  'pcb3',
  'pcb4',
  'pipe_fryum'
];

export default function HistoryPage() {
  const [inspections, setInspections] = useState([]);
  const [category, setCategory] = useState('all');
  const [statusFilter, setStatusFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [selectedRecord, setSelectedRecord] = useState(null);

  const loadData = async () => {
    setLoading(true);
    try {
      const data = await fetchInspections(category, statusFilter, 200);
      setInspections(data);
    } catch (err) {
      console.error('History load error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [category, statusFilter]);

  return (
    <div className="page-content">
      <div className="page-title-section">
        <h1 className="page-title">Inspection Audit History</h1>
        <p className="page-subtitle">Persistent MongoDB historical inspection log with filtering</p>
      </div>

      {/* Filters */}
      <div className="filter-bar">
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="custom-select"
        >
          {VISA_CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c === 'all' ? 'All Categories' : c.toUpperCase()}
            </option>
          ))}
        </select>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          className="custom-select"
        >
          <option value="all">All Statuses</option>
          <option value="NORMAL">NORMAL</option>
          <option value="ANOMALY">ANOMALY</option>
        </select>

        <button
          onClick={loadData}
          style={{
            backgroundColor: '#334155',
            color: '#fff',
            border: 'none',
            padding: '0.6rem 1rem',
            borderRadius: '8px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem'
          }}
        >
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      {/* History Table */}
      {loading ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8' }}>Loading inspection records...</div>
      ) : inspections.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '3rem', color: '#94a3b8', backgroundColor: '#1e293b', borderRadius: '12px' }}>
          No inspection records matching filters. Run inspections in the Inspect tab to populate history logs.
        </div>
      ) : (
        <table className="custom-table">
          <thead>
            <tr>
              <th>Inspection ID</th>
              <th>Category</th>
              <th>Status</th>
              <th>Anomaly Score</th>
              <th>Threshold</th>
              <th>Defect Area %</th>
              <th>Timestamp</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {inspections.map((item) => (
              <tr key={item.inspection_id}>
                <td style={{ fontWeight: 600 }}>{item.inspection_id}</td>
                <td style={{ textTransform: 'capitalize' }}>{item.category}</td>
                <td>
                  <span className={`badge-status ${item.status === 'NORMAL' ? 'badge-normal' : 'badge-anomaly'}`}>
                    {item.status === 'NORMAL' ? <CheckCircle size={14} /> : <AlertTriangle size={14} />}
                    {item.status}
                  </span>
                </td>
                <td style={{ fontWeight: 600, color: item.status === 'ANOMALY' ? '#f87171' : '#34d399' }}>
                  {item.anomaly_score}
                </td>
                <td>{item.threshold}</td>
                <td>{item.defect_area_percentage}%</td>
                <td style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                  {new Date(item.timestamp).toLocaleString()}
                </td>
                <td>
                  <button
                    onClick={() => setSelectedRecord(item)}
                    style={{
                      backgroundColor: '#3b82f620',
                      color: '#60a5fa',
                      border: '1px solid #3b82f640',
                      padding: '0.35rem 0.75rem',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      fontSize: '0.8rem'
                    }}
                  >
                    <Eye size={14} /> Inspect
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Record Modal */}
      {selectedRecord && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100
          }}
          onClick={() => setSelectedRecord(null)}
        >
          <div
            style={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '14px',
              padding: '2rem',
              maxWidth: '800px',
              width: '90%',
              maxHeight: '90vh',
              overflowY: 'auto'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.25rem' }}>Inspection Record: {selectedRecord.inspection_id}</h3>
              <button onClick={() => setSelectedRecord(null)} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: '1.2rem' }}>✕</button>
            </div>
            {selectedRecord.image_url && (
              <img src={selectedRecord.image_url} alt="Heatmap preview" style={{ width: '100%', borderRadius: '8px', marginBottom: '1rem' }} />
            )}
            <div style={{ fontSize: '0.9rem', color: '#94a3b8' }}>
              <p>Category: <strong style={{ color: '#fff', textTransform: 'capitalize' }}>{selectedRecord.category}</strong></p>
              <p>Status: <strong style={{ color: selectedRecord.status === 'ANOMALY' ? '#f87171' : '#34d399' }}>{selectedRecord.status}</strong></p>
              <p>Anomaly Score: {selectedRecord.anomaly_score} (Threshold: {selectedRecord.threshold})</p>
              <p>Defect Area: {selectedRecord.defect_area_percentage}%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

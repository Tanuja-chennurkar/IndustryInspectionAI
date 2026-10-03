import React, { useState } from 'react';
import { Upload, Scan, CheckCircle, AlertTriangle, Cpu, Clock } from 'lucide-react';
import { inspectImage } from '../services/api';

const VISA_CATEGORIES = [
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

export default function InspectPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [category, setCategory] = useState('');
  const [customThreshold, setCustomThreshold] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please select a valid JPEG or PNG image file.');
      return;
    }
    setError(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInspect = async () => {
    if (!selectedFile) {
      setError('Please upload an image to inspect.');
      return;
    }
    if (!category) {
      setError('Please select a product category before running the inspection.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await inspectImage(selectedFile, category, customThreshold);
      setResult(res);
    } catch (err) {
      setError(err.message || 'Inspection failed. Ensure backend API is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-content">
      <div className="page-title-section">
        <div>
          <h1 className="page-title">Image Inspection</h1>
          <p className="page-subtitle">Choose a product category and inspect an image for anomalies.</p>
        </div>
      </div>

        <div className="inspection-workspace">
          {/* Drag & Drop Upload Zone */}
          <div
            className={`dropzone-container ${isDragOver ? 'drag-over' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-upload-input').click()}
          >
            <input
              id="file-upload-input"
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => e.target.files[0] && handleFileSelect(e.target.files[0])}
            />
            <Upload className="dropzone-icon" />
            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>
              {selectedFile ? selectedFile.name : 'Drag & Drop Product Image Here'}
            </h3>
            <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
              {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : 'Supports PNG, JPG, JPEG up to 10MB'}
            </p>
          </div>

          {/* Controls Bar */}
          <div className="filter-bar" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <div>
                <label htmlFor="inspection-category" className="inspection-label">Product category</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="custom-select"
                >
                  <option value="" disabled>Select a category</option>
                  {VISA_CATEGORIES.map((cat) => (
                    <option key={cat} value={cat}>
                      {cat.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="inspection-threshold" className="inspection-label">Custom threshold <span>Optional</span></label>
                <input
                  id="inspection-threshold"
                  type="number"
                  step="0.01"
                  placeholder="Model default"
                  value={customThreshold}
                  onChange={(e) => setCustomThreshold(e.target.value)}
                  className="custom-input"
                />
              </div>
            </div>

            <button
              onClick={handleInspect}
              disabled={loading || !selectedFile || !category}
              style={{
                backgroundColor: loading ? '#334155' : '#3b82f6',
                color: '#fff',
                border: 'none',
                padding: '0.75rem 2rem',
                borderRadius: '10px',
                fontWeight: 700,
                cursor: loading || !selectedFile || !category ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                fontSize: '1rem'
              }}
            >
              <Scan size={20} />
              {loading ? 'Executing TensorFlow Inference...' : 'Run Inspection'}
            </button>
          </div>
        </div>

          {error && (
            <div style={{ backgroundColor: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', color: '#f87171', padding: '1rem', borderRadius: '10px', marginTop: '1rem' }}>
              {error}
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="result-grid">
              <div className="panel-card">
                <div className="panel-header">
                  <span>Inspection result</span>
                  <span className={`badge-status ${result.status === 'NORMAL' ? 'badge-normal' : 'badge-anomaly'}`}>
                    {result.status === 'NORMAL' ? <CheckCircle size={16} /> : <AlertTriangle size={16} />}
                    {result.status}
                  </span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ backgroundColor: '#0f172a', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Inspection ID</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{result.inspection_id}</div>
                  </div>

                  <div style={{ backgroundColor: '#0f172a', padding: '1rem', borderRadius: '8px' }}>
                      <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Selected category</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', textTransform: 'capitalize' }}>
                        {result.category.replaceAll('_', ' ')}
                    </div>
                  </div>

                  <div style={{ backgroundColor: '#0f172a', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Anomaly Score</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem', color: result.anomaly_score >= result.threshold ? '#f87171' : '#34d399' }}>
                      {result.anomaly_score} <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>(Thresh: {result.threshold})</span>
                    </div>
                  </div>

                  <div style={{ backgroundColor: '#0f172a', padding: '1rem', borderRadius: '8px' }}>
                    <div style={{ color: '#94a3b8', fontSize: '0.8rem' }}>Defect Area %</div>
                    <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{result.defect_area_percentage}%</div>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '1.5rem', color: '#94a3b8', fontSize: '0.85rem' }}>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Clock size={16} /> Latency: {result.processing_time_ms} ms
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                    <Cpu size={16} /> Version: {result.model_version}
                  </span>
                </div>

                {result.defect_regions && result.defect_regions.length > 0 && (
                  <div style={{ marginTop: '1.5rem' }}>
                    <h4 style={{ fontSize: '0.9rem', marginBottom: '0.5rem', color: '#f8fafc' }}>Detected Defect Region Coordinates</h4>
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Region #</th>
                          <th>BBox [X, Y, W, H]</th>
                          <th>Area (px)</th>
                          <th>Centroid</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.defect_regions.map((reg, idx) => (
                          <tr key={idx}>
                            <td>#{idx + 1}</td>
                            <td>[{reg.bbox.join(', ')}]</td>
                            <td>{reg.area_pixels}</td>
                            <td>({reg.centroid.join(', ')})</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="panel-card">
                <div className="panel-header">
                  <span>Heatmap visualization</span>
                </div>
                {result.image_url ? (
                  <img src={result.image_url} alt="Anomaly Heatmap Visualization" className="heatmap-preview" />
                ) : previewUrl ? (
                  <img src={previewUrl} alt="Uploaded Image" className="heatmap-preview" />
                ) : null}
              </div>
            </div>
          )}
    </div>
  );
}

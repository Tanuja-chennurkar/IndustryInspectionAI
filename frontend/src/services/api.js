/**
 * REST API Client Service connecting React Dashboard to FastAPI Backend.
 */

const API_BASE_URL = '/api';

export async function inspectImage(file, category = '', threshold = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (category) formData.append('category', category);
  if (threshold !== null && threshold !== '') formData.append('threshold', threshold);

  const response = await fetch(`${API_BASE_URL}/inspect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Inspection request failed');
  }

  return response.json();
}

export async function fetchInspections(category = '', status = '', limit = 100) {
  const params = new URLSearchParams();
  if (category && category !== 'all') params.append('category', category);
  if (status && status !== 'all') params.append('status', status);
  params.append('limit', limit);

  const response = await fetch(`${API_BASE_URL}/inspections?${params.toString()}`);
  if (!response.ok) throw new Error('Failed to fetch inspection history');
  return response.json();
}

export async function fetchInspectionById(inspectionId) {
  const response = await fetch(`${API_BASE_URL}/inspections/${inspectionId}`);
  if (!response.ok) throw new Error(`Failed to fetch inspection ${inspectionId}`);
  return response.json();
}

export async function fetchStatistics() {
  const response = await fetch(`${API_BASE_URL}/statistics`);
  if (!response.ok) throw new Error('Failed to fetch inspection statistics');
  return response.json();
}

export async function fetchDefects(limit = 50) {
  const response = await fetch(`${API_BASE_URL}/defects?limit=${limit}`);
  if (!response.ok) throw new Error('Failed to fetch defects log');
  return response.json();
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE_URL}/models`);
  if (!response.ok) throw new Error('Failed to fetch model metadata');
  return response.json();
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) throw new Error('Failed to fetch backend health status');
  return response.json();
}

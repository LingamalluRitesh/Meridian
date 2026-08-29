const API_BASE = 'http://localhost:8000';

export async function fetchApi<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string> || {})
    }
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export const api = {
  getHealth: () => fetchApi('/health'),
  getModels: () => fetchApi('/api/v1/models'),
  predict: (modelId: string, features: any[]) => fetchApi('/api/v1/predict', { method: 'POST', body: JSON.stringify({ model_id: modelId, features }) }),
  getModelCards: () => fetchApi('/api/v1/governance/model-cards')
};

const API_BASE_URL = 'http://127.0.0.1:8000'

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error('API server is not responding.')
  }
  return response.json()
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE_URL}/models`)
  if (!response.ok) {
    throw new Error('Failed to fetch available models and gates.')
  }
  return response.json()
}

export async function fetchTasks(split = 'val', limit = 20, offset = 0, search = '') {
  const params = new URLSearchParams({ split, limit: String(limit), offset: String(offset) })
  if (search) {
    params.append('search', search)
  }
  const response = await fetch(`${API_BASE_URL}/tasks?${params.toString()}`)
  if (!response.ok) {
    throw new Error('Failed to load benchmark tasks.')
  }
  return response.json()
}

export async function fetchHistory() {
  const response = await fetch(`${API_BASE_URL}/history`)
  if (!response.ok) {
    throw new Error('Failed to load execution history.')
  }
  return response.json()
}

export async function clearHistory() {
  const response = await fetch(`${API_BASE_URL}/history`, { method: 'DELETE' })
  if (!response.ok) {
    throw new Error('Failed to clear execution history.')
  }
  return response.json()
}

export async function runGateOrchestra(payload) {
  const response = await fetch(`${API_BASE_URL}/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  let data = null
  try {
    data = await response.json()
  } catch {
    // The error response may not contain JSON.
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || 'Request failed.'
    throw new Error(typeof detail === 'string' ? detail : 'Request failed.')
  }

  return data
}

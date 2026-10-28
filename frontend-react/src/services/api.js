// Override with VITE_API_BASE_URL (e.g. in frontend-react/.env.local).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export async function checkHealth() {
  let response
  try {
    response = await fetch(`${API_BASE_URL}/`)
  } catch {
    throw new Error(
      `No GateOrchestra API at ${API_BASE_URL}. Start it with: uvicorn api.main:app --port 8000`,
    )
  }
  const data = response.ok ? await response.json().catch(() => null) : null
  // Another local app can hold the port; make sure this is really our backend.
  if (data?.name !== 'GateOrchestra API') {
    throw new Error(
      `The server at ${API_BASE_URL} is not the GateOrchestra API (another app is using that port). ` +
        'Run the API on a free port and set VITE_API_BASE_URL in frontend-react/.env.local.',
    )
  }
  return data
}

export async function fetchModels() {
  const response = await fetch(`${API_BASE_URL}/models`)
  if (!response.ok) {
    throw new Error('Failed to fetch available models and gates.')
  }
  return response.json()
}

export async function fetchBenchmarkSummary() {
  const response = await fetch(`${API_BASE_URL}/demo/summary`)
  if (!response.ok) {
    throw new Error('Failed to load benchmark results.')
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

export async function runGateOrchestra(payload, { signal } = {}) {
  const response = await fetch(`${API_BASE_URL}/run`, {
    method: 'POST',
    signal,
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

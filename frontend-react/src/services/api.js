const API_BASE_URL = 'http://127.0.0.1:8000'

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
    data = null
  }

  if (!response.ok) {
    const detail = data?.detail || data?.message || 'Request failed.'
    throw new Error(typeof detail === 'string' ? detail : 'Request failed.')
  }

  return data
}

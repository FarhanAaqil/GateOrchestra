import type {
  ImportanceResponse,
  ModelsResponse,
  ResultsDetail,
  RunRequest,
  RunResult,
  SummaryResponse,
  TasksResponse,
} from './types'

// Override with VITE_API_BASE_URL (e.g. in frontend-react/.env.local).
export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  let data: unknown = null
  try {
    data = await response.json()
  } catch {
    // Some error responses carry no JSON body.
  }
  if (!response.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return data as T
}

/** Confirms the server on API_BASE_URL is really GateOrchestra, not another local app. */
export async function checkBackend(): Promise<void> {
  let root: { name?: string } | null
  try {
    root = await request<{ name?: string }>('/')
  } catch {
    throw new Error(`No GateOrchestra API at ${API_BASE_URL}. Start it with: uvicorn api.main:app`)
  }
  if (root?.name !== 'GateOrchestra API') {
    throw new Error(
      `The server at ${API_BASE_URL} is not the GateOrchestra API (another app holds that port). ` +
        'Run the API on a free port and set VITE_API_BASE_URL in frontend-react/.env.local.',
    )
  }
}

export const api = {
  models: () => request<ModelsResponse>('/models'),
  tasks: (split: 'train' | 'val' | 'test', limit = 100, offset = 0) =>
    request<TasksResponse>(`/tasks?split=${split}&limit=${limit}&offset=${offset}`),
  history: () => request<{ count: number; history: RunResult[] }>('/history'),
  summary: () => request<SummaryResponse>('/demo/summary'),
  gateImportance: () => request<ImportanceResponse>('/gate/importance'),
  resultsDetail: () => request<ResultsDetail>('/results/detail'),
  run: (body: RunRequest, signal?: AbortSignal) =>
    request<RunResult>('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    }),
}

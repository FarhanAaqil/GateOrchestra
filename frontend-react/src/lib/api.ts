import { AuthRequired, authHeaders } from './auth'
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
  const response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers: authHeaders(init?.headers) })
  if (response.status === 401) throw new AuthRequired()
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

// GET responses are cached briefly and concurrent identical requests share one
// fetch, so pages that need the same data (e.g. /models, /demo/summary) do not
// each hit the API. Reloads and runs clear the cache.
const CACHE_TTL_MS = 30_000
const cache = new Map<string, { at: number; value: Promise<unknown> }>()

function cachedGet<T>(path: string): Promise<T> {
  const hit = cache.get(path)
  if (hit && Date.now() - hit.at < CACHE_TTL_MS) return hit.value as Promise<T>
  const value = request<T>(path)
  cache.set(path, { at: Date.now(), value })
  // Never keep a failure: the next call should retry.
  value.catch(() => {
    if (cache.get(path)?.value === value) cache.delete(path)
  })
  return value
}

/** Drop every cached GET response (used by reload buttons and after a run). */
export function clearApiCache(): void {
  cache.clear()
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

export interface BenchmarkRun {
  id: string
  label: string
  seeds: number[]
  main: boolean
}

export const api = {
  models: () => cachedGet<ModelsResponse>('/models'),
  tasks: (split: 'train' | 'val' | 'test', limit = 100, offset = 0) =>
    cachedGet<TasksResponse>(`/tasks?split=${split}&limit=${limit}&offset=${offset}`),
  history: () => request<{ count: number; history: RunResult[] }>('/history'),
  summary: (run?: string) => cachedGet<SummaryResponse>(run && run !== 'final' ? `/demo/summary?run=${encodeURIComponent(run)}` : '/demo/summary'),
  runs: () => cachedGet<{ runs: BenchmarkRun[] }>('/results/runs'),
  gateImportance: () => cachedGet<ImportanceResponse>('/gate/importance'),
  resultsDetail: (run?: string) => cachedGet<ResultsDetail>(run && run !== 'final' ? `/results/detail?run=${encodeURIComponent(run)}` : '/results/detail'),
  run: async (body: RunRequest, signal?: AbortSignal) => {
    const result = await request<RunResult>('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal,
    })
    clearApiCache()
    return result
  },
}

import { API_BASE_URL } from './api'
import { configureSession, withAuth } from './auth'

/** The signed-in user, as /auth/me returns it (hosted mode only). */
export interface SessionUser {
  id: string
  email: string
  verified: boolean
  is_admin: boolean
  plan: string
  paid_until: number | null
  created_at: number
}

interface SessionReply {
  user: SessionUser
  csrf_token: string
}

async function call<T>(path: string, init: RequestInit = {}): Promise<{ status: number; data: T }> {
  const response = await fetch(`${API_BASE_URL}${path}`, withAuth(init))
  const data = (await response.json().catch(() => null)) as T
  if (!response.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail
    const error = new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
    ;(error as Error & { status?: number }).status = response.status
    throw error
  }
  return { status: response.status, data }
}

function post<T>(path: string, body?: unknown) {
  return call<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

export const authApi = {
  /** The current session, or null when nobody is signed in. */
  me: async (): Promise<SessionUser | null> => {
    try {
      const { data } = await call<SessionReply>('/auth/me')
      configureSession(true, data.csrf_token)
      return data.user
    } catch (err) {
      if ((err as { status?: number }).status === 401) return null
      throw err
    }
  },
  login: async (email: string, password: string): Promise<SessionUser> => {
    const { data } = await post<SessionReply>('/auth/login', { email, password })
    configureSession(true, data.csrf_token)
    return data.user
  },
  logout: async (): Promise<void> => {
    try {
      await post('/auth/logout')
    } finally {
      configureSession(true, '')
    }
  },
  signup: (email: string, password: string) => post('/auth/signup', { email, password }),
  verify: (token: string) => post('/auth/verify', { token }),
  resendVerification: () => post('/auth/resend-verification'),
  forgot: (email: string) => post('/auth/forgot', { email }),
  reset: (token: string, password: string) => post('/auth/reset', { token, password }),
}

export const MIN_PASSWORD = 10

export interface OwnKey {
  provider: 'groq' | 'gemini'
  added_at: number
}

function send<T>(path: string, method: string, body?: unknown) {
  return call<T>(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => r.data)
}

/** The signed-in person's own settings (hosted mode). */
export const accountApi = {
  keys: () => call<{ keys: OwnKey[] }>('/account/keys').then((r) => r.data),
  putKey: (provider: 'groq' | 'gemini', key: string) => send<{ keys: OwnKey[] }>('/account/keys', 'PUT', { provider, key }),
  deleteKey: (provider: string) => send<{ keys: OwnKey[] }>(`/account/keys/${provider}`, 'DELETE'),
}

export interface PlanInfo {
  id: string
  label: string
  price_usd_month: number
  messages_per_day: number
  documents: number
  daily_tokens: number
  concurrent_chats: number
  blurb: string
}

export interface UsageCounts {
  messages: number
  tokens: number
  own_key_tokens: number
  direct: number
  escalated: number
  small_talk: number
  cached: number
}

export interface UsageReport {
  plan: PlanInfo
  paid_until: number | null
  today: UsageCounts
  month: UsageCounts
  documents: number
  own_key: boolean
  estimate: {
    available: boolean
    sample_size: number
    per_escalation: number | null
    saved_tokens: number | null
    window_days: number
    method: string
  }
}

export const usageApi = {
  usage: () => call<UsageReport>('/account/usage').then((r) => r.data),
  plans: () => call<{ default: string; plans: PlanInfo[] }>('/plans').then((r) => r.data),
}

export interface AdminUser extends SessionUser {
  effective_plan: string
  today: UsageCounts
}

export const billingApi = {
  /** The payment page for the signed-in person (404 when payments are not set up). */
  upgrade: () => call<{ url: string }>('/account/upgrade').then((r) => r.data),
  plans: () => call<{ default: string; plans: PlanInfo[]; payments: boolean }>('/plans').then((r) => r.data),
}

export const adminApi = {
  users: (q = '') => call<{ users: AdminUser[] }>(`/admin/users?q=${encodeURIComponent(q)}`).then((r) => r.data),
  setPlan: (id: string, plan: string, days: number) => send<{ user: AdminUser }>(`/admin/users/${id}/plan`, 'POST', { plan, days }),
}

export const profileApi = {
  /** Everything of yours as a JSON file download. */
  exportData: async (): Promise<Blob> => {
    const response = await fetch(`${API_BASE_URL}/account/export`, withAuth())
    if (!response.ok) throw new Error(`Export failed (${response.status})`)
    return response.blob()
  },
  changePassword: async (current: string, next: string): Promise<void> => {
    const { data } = await call<SessionReply>('/account/password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ current, new: next }),
    })
    configureSession(true, data.csrf_token)
  },
  deleteAccount: (password: string) => send<{ status: string }>('/account', 'DELETE', { password }),
}

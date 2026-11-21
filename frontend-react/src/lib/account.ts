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
  two_factor?: boolean
}

export type LoginResult = { user: SessionUser; challenge?: undefined } | { challenge: string; user?: undefined }

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
  /** A session, or a challenge when the account uses two-factor sign-in. */
  login: async (email: string, password: string): Promise<LoginResult> => {
    const { data } = await post<SessionReply | { two_factor_required: true; challenge: string }>('/auth/login', { email, password })
    if ('two_factor_required' in data) return { challenge: data.challenge }
    configureSession(true, data.csrf_token)
    return { user: data.user }
  },
  loginSecondFactor: async (challenge: string, code: string): Promise<SessionUser> => {
    const { data } = await post<SessionReply>('/auth/login/2fa', { challenge, code })
    configureSession(true, data.csrf_token)
    return data.user
  },
  confirmEmail: (token: string) => post<{ status: string; email: string }>('/auth/confirm-email', { token }).then((r) => r.data),
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

export type KeyProvider = 'groq' | 'gemini' | 'openai'

export interface OwnKey {
  provider: KeyProvider
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
  putKey: (provider: KeyProvider, key: string) => send<{ keys: OwnKey[] }>('/account/keys', 'PUT', { provider, key }),
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
    source: 'server' | 'evaluation' | null
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

export const securityApi = {
  status: () => call<{ two_factor: boolean; recovery_codes_left: number }>('/account/security').then((r) => r.data),
  setup: (password: string) =>
    call<{ secret: string; otpauth_uri: string }>('/account/2fa/setup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password }),
    }).then((r) => r.data),
  enable: (code: string) => send<{ recovery_codes: string[] }>('/account/2fa/enable', 'POST', { code }),
  disable: (password: string, code: string) => send<{ status: string }>('/account/2fa/disable', 'POST', { password, code }),
  changeEmail: (newEmail: string, password: string) => send<{ status: string }>('/account/email', 'POST', { new_email: newEmail, password }),
}

/**
 * The API token for a deployed GateOrchestra (GATE_API_TOKEN on the server).
 *
 * It is typed in by the user and kept only in this tab's sessionStorage, never
 * baked into the build, so the published bundle carries no secret.
 */
const TOKEN_KEY = 'gateorchestra.apiToken'

export class AuthRequired extends Error {
  constructor() {
    super('This GateOrchestra server needs an API token.')
    this.name = 'AuthRequired'
  }
}

export function getToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? ''
  } catch {
    return ''
  }
}

export function setToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token)
    else sessionStorage.removeItem(TOKEN_KEY)
  } catch {
    // Storage may be blocked; the token then lasts only for this page.
  }
}

export function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra)
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return headers
}

/**
 * Hosted mode: users sign in with an httpOnly session cookie the page cannot read.
 * Every write must also carry the session's CSRF token, which the API hands out at
 * login and from /auth/me; it is kept in memory only.
 */
const session = { hosted: false, csrf: '' }

export function configureSession(hosted: boolean, csrf = ''): void {
  session.hosted = hosted
  session.csrf = csrf
}

export function isHosted(): boolean {
  return session.hosted
}

/** fetch() options with the token or session headers (and the cookie when hosted). */
export function withAuth(init: RequestInit = {}): RequestInit {
  const headers = authHeaders(init.headers)
  if (session.hosted && session.csrf) headers.set('X-CSRF-Token', session.csrf)
  return session.hosted ? { ...init, headers, credentials: 'include' } : { ...init, headers }
}

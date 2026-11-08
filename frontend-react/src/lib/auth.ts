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

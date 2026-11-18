import { createContext, useContext } from 'react'
import type { SessionUser } from './account'
import type { ModelsResponse, RunResult } from './types'

export interface AppState {
  backend: 'checking' | 'online' | 'error' | 'locked'
  backendError: string | null
  models: ModelsResponse | null
  /** Hosted (multi-user) server: people sign in, and operator pages are admin-only. */
  hosted: boolean
  /** The signed-in user (hosted mode); null when signed out or single-tenant. */
  user: SessionUser | null
  setUser: (user: SessionUser | null) => void
  /** True once the session check has finished (so guards do not redirect too early). */
  sessionChecked: boolean
  /** Single-tenant servers have no accounts, so everyone there is the operator. */
  isAdmin: boolean
  /** Most recent run from this browser session (shared by Run and Gate Inspector). */
  lastRun: RunResult | null
  setLastRun: (run: RunResult) => void
  /** Learned-gate ESCALATE threshold sent with runs; null = classifier default. */
  threshold: number | null
  setThreshold: (value: number | null) => void
  performanceMode: boolean
  setPerformanceMode: (value: boolean) => void
  reducedMotion: boolean
  isMobile: boolean
  /** True when the full 3D scenes should render. */
  use3D: boolean
}

export const AppContext = createContext<AppState | null>(null)

export function useApp(): AppState {
  const value = useContext(AppContext)
  if (!value) throw new Error('useApp must be used inside <AppProvider>')
  return value
}

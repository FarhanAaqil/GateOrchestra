import { createContext, useContext } from 'react'
import type { ModelsResponse, RunResult } from './types'

export interface AppState {
  backend: 'checking' | 'online' | 'error'
  backendError: string | null
  models: ModelsResponse | null
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

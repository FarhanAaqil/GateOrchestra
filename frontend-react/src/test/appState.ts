import type { AppState } from '../lib/context'

/** A complete AppState for component tests; override what the test is about. */
export function appState(overrides: Partial<AppState> = {}): AppState {
  return {
    backend: 'online',
    backendError: null,
    models: null,
    hosted: true,
    user: null,
    setUser: () => undefined,
    sessionChecked: true,
    isAdmin: false,
    lastRun: null,
    setLastRun: () => undefined,
    threshold: null,
    setThreshold: () => undefined,
    performanceMode: true,
    setPerformanceMode: () => undefined,
    reducedMotion: true,
    isMobile: false,
    use3D: false,
    ...overrides,
  }
}

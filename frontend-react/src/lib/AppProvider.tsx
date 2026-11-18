import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { authApi, type SessionUser } from './account'
import { api, checkBackend } from './api'
import { AuthRequired, configureSession } from './auth'
import { AppContext, type AppState } from './context'
import type { ModelsResponse, RunResult } from './types'

const PERF_KEY = 'gateorchestra.performanceMode'

function readStoredPerformanceMode(): boolean {
  try {
    return localStorage.getItem(PERF_KEY) === '1'
  } catch {
    return false
  }
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => window.matchMedia(query).matches)
  useEffect(() => {
    const list = window.matchMedia(query)
    const onChange = () => setMatches(list.matches)
    list.addEventListener('change', onChange)
    return () => list.removeEventListener('change', onChange)
  }, [query])
  return matches
}

function webglAvailable(): boolean {
  try {
    const canvas = document.createElement('canvas')
    return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'))
  } catch {
    return false
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [backend, setBackend] = useState<AppState['backend']>('checking')
  const [backendError, setBackendError] = useState<string | null>(null)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [hosted, setHosted] = useState(false)
  const [user, setUser] = useState<SessionUser | null>(null)
  const [sessionChecked, setSessionChecked] = useState(false)
  const [lastRun, setLastRun] = useState<RunResult | null>(null)
  const [threshold, setThreshold] = useState<number | null>(null)
  const [performanceMode, setPerformanceModeState] = useState(readStoredPerformanceMode)
  const reducedMotion = useMediaQuery('(prefers-reduced-motion: reduce)')
  const isMobile = useMediaQuery('(max-width: 767px)')
  const [hasWebgl] = useState(webglAvailable)

  useEffect(() => {
    let cancelled = false
    checkBackend()
      .then(async (root) => {
        configureSession(root.hosted)
        if (cancelled) return
        setHosted(root.hosted)
        if (root.hosted) {
          const current = await authApi.me().catch(() => null)
          if (!cancelled) setUser(current)
        }
        if (!cancelled) setSessionChecked(true)
      })
      .then(() => api.models())
      .then((data) => {
        if (cancelled) return
        setModels(data)
        setBackend('online')
      })
      .catch((err: Error) => {
        if (cancelled) return
        setBackend(err instanceof AuthRequired ? 'locked' : 'error')
        setBackendError(err.message)
        setSessionChecked(true)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const value = useMemo<AppState>(
    () => ({
      backend,
      backendError,
      models,
      hosted,
      user,
      setUser,
      sessionChecked,
      isAdmin: !hosted || Boolean(user?.is_admin),
      lastRun,
      setLastRun,
      threshold,
      setThreshold,
      performanceMode,
      setPerformanceMode: (next: boolean) => {
        setPerformanceModeState(next)
        try {
          localStorage.setItem(PERF_KEY, next ? '1' : '0')
        } catch {
          // Storage may be unavailable (private mode); the toggle still works.
        }
      },
      reducedMotion,
      isMobile,
      use3D: hasWebgl && !performanceMode && !reducedMotion && !isMobile,
    }),
    [backend, backendError, models, hosted, user, sessionChecked, lastRun, threshold, performanceMode, reducedMotion, isMobile, hasWebgl],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

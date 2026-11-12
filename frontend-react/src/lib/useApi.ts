import { useCallback, useEffect, useState } from 'react'
import { clearApiCache } from './api'

export interface ApiState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/**
 * Fetch on mount, on reload, and whenever `key` changes; keeps the last good data while
 * reloading. Pass a key when the fetcher depends on a value, e.g. the selected run.
 */
export function useApi<T>(fetcher: () => Promise<T>, key?: unknown): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let cancelled = false
    fetcher()
      .then((value) => {
        if (cancelled) return
        setData(value)
        setError(null)
      })
      .catch((err: Error) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // The fetcher is expected to be stable (module-level api functions).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, key])

  const reload = useCallback(() => {
    clearApiCache()
    setLoading(true)
    setNonce((n) => n + 1)
  }, [])

  return { data, error, loading, reload }
}

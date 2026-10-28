import { useCallback, useEffect, useState } from 'react'

export interface ApiState<T> {
  data: T | null
  error: string | null
  loading: boolean
  reload: () => void
}

/** Fetch once on mount (and on reload); keeps the last good data while reloading. */
export function useApi<T>(fetcher: () => Promise<T>): ApiState<T> {
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
  }, [nonce])

  const reload = useCallback(() => {
    setLoading(true)
    setNonce((n) => n + 1)
  }, [])

  return { data, error, loading, reload }
}

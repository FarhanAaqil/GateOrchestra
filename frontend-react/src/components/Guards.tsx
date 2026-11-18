import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { useApp } from '../lib/context'
import { Spinner } from './ui'

/** Hosted mode: send signed-out visitors to the login page, then back here. */
export function RequireUser({ children }: { children: ReactNode }) {
  const { hosted, user, sessionChecked } = useApp()
  const location = useLocation()
  if (!hosted) return <>{children}</>
  if (!sessionChecked) return <Spinner label="Checking your session" />
  if (!user) return <Navigate to={`/login?next=${encodeURIComponent(location.pathname + location.search)}`} replace />
  return <>{children}</>
}

/**
 * Operator pages. In hosted mode anyone who is not an admin is sent home, as if the
 * page did not exist; the API refuses their requests as well, so hiding is not the
 * only protection.
 */
export function AdminOnly({ children }: { children: ReactNode }) {
  const { hosted, isAdmin, sessionChecked } = useApp()
  if (!hosted) return <>{children}</>
  if (!sessionChecked) return <Spinner label="Checking your session" />
  if (!isAdmin) return <Navigate to="/" replace />
  return <>{children}</>
}

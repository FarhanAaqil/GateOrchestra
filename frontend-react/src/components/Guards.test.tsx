import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { SessionUser } from '../lib/account'
import { AppContext, type AppState } from '../lib/context'
import { AdminOnly, RequireUser } from './Guards'

const USER: SessionUser = { id: 'u1', email: 'ada@example.com', verified: true, is_admin: false, plan: 'free', paid_until: null, created_at: 0 }

function state(overrides: Partial<AppState>): AppState {
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

function renderAt(path: string, value: AppState) {
  return render(
    <AppContext.Provider value={value}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/" element={<p>home page</p>} />
          <Route path="/login" element={<p>login page</p>} />
          <Route path="/admin" element={<AdminOnly><p>admin page</p></AdminOnly>} />
          <Route path="/chat" element={<RequireUser><p>chat page</p></RequireUser>} />
        </Routes>
      </MemoryRouter>
    </AppContext.Provider>,
  )
}

describe('route guards', () => {
  it('sends a normal user away from the admin page when hosted', () => {
    renderAt('/admin', state({ user: USER }))
    expect(screen.getByText('home page')).toBeInTheDocument()
    expect(screen.queryByText('admin page')).toBeNull()
  })

  it('lets an admin in', () => {
    renderAt('/admin', state({ user: { ...USER, is_admin: true }, isAdmin: true }))
    expect(screen.getByText('admin page')).toBeInTheDocument()
  })

  it('keeps the single-tenant console unchanged', () => {
    renderAt('/admin', state({ hosted: false, isAdmin: true }))
    expect(screen.getByText('admin page')).toBeInTheDocument()
    renderAt('/chat', state({ hosted: false }))
    expect(screen.getByText('chat page')).toBeInTheDocument()
  })

  it('asks signed-out visitors to log in before chatting', () => {
    renderAt('/chat', state({}))
    expect(screen.getByText('login page')).toBeInTheDocument()
  })

  it('waits for the session check instead of redirecting early', () => {
    renderAt('/chat', state({ sessionChecked: false }))
    expect(screen.queryByText('login page')).toBeNull()
  })
})

import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import type { SessionUser } from '../lib/account'
import { AppContext, type AppState } from '../lib/context'
import { appState } from '../test/appState'
import Layout from './Layout'

const USER: SessionUser = { id: 'u1', email: 'ada@example.com', verified: true, is_admin: false, plan: 'free', paid_until: null, created_at: 0 }

function menu(value: AppState): string[] {
  render(
    <AppContext.Provider value={value}>
      <MemoryRouter initialEntries={['/about']}>
        <Layout>
          <p>page</p>
        </Layout>
      </MemoryRouter>
    </AppContext.Provider>,
  )
  const nav = screen.getAllByRole('navigation', { name: 'Main' })[0]
  return within(nav).getAllByRole('link').map((a) => a.textContent ?? '')
}

describe('main menu', () => {
  it('hides operator pages from a signed-out visitor on a hosted server', () => {
    const items = menu(appState())
    for (const hidden of ['Admin', 'Run a Task', 'Token Accountant', 'Settings', 'Usage']) expect(items).not.toContain(hidden)
    expect(items).toEqual(expect.arrayContaining(['Chat', 'Results', 'Pricing', 'About']))
  })

  it('shows account pages, not operator pages, to a normal user', () => {
    const items = menu(appState({ user: USER }))
    expect(items).toEqual(expect.arrayContaining(['Usage', 'Settings']))
    expect(items).not.toContain('Admin')
  })

  it('shows everything to an admin', () => {
    const items = menu(appState({ user: { ...USER, is_admin: true }, isAdmin: true }))
    expect(items).toEqual(expect.arrayContaining(['Admin', 'Run a Task', 'Token Accountant']))
  })

  it('keeps the single-tenant console unchanged', () => {
    const items = menu(appState({ hosted: false, isAdmin: true }))
    expect(items).toEqual(expect.arrayContaining(['Admin', 'Run a Task', 'Overview']))
    expect(items).not.toContain('Pricing')
  })

  it('asks an unconfirmed user to confirm their email', () => {
    menu(appState({ user: { ...USER, verified: false } }))
    expect(screen.getByText(/with the link we emailed you/)).toBeInTheDocument()
  })
})

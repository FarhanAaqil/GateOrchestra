import { AnimatePresence, motion } from 'framer-motion'
import {
  BarChart3,
  Box,
  ChartPie,
  Coins,
  CreditCard,
  Database,
  FlaskConical,
  Gauge,
  Info,
  LayoutDashboard,
  LogIn,
  LogOut,
  Menu,
  MessagesSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  ServerCog,
  Settings as SettingsIcon,
  X,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { authApi } from '../lib/account'
import { setToken } from '../lib/auth'
import { chatApi } from '../lib/chat'
import { useApp } from '../lib/context'

interface NavItem {
  to: string
  label: string
  icon: typeof Info
  adminOnly?: boolean
  /** Only for a signed-in person on a hosted server. */
  accountOnly?: boolean
  /** Only on a hosted server. */
  hostedOnly?: boolean
}

const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: 'Assistant',
    items: [
      { to: '/chat', label: 'Chat', icon: MessagesSquare },
      { to: '/usage', label: 'Usage', icon: ChartPie, accountOnly: true },
      { to: '/run', label: 'Run a Task', icon: Play, adminOnly: true },
    ],
  },
  {
    label: 'Research',
    items: [
      { to: '/', label: 'Overview', icon: LayoutDashboard },
      { to: '/results', label: 'Results', icon: FlaskConical },
      { to: '/benchmark', label: 'Benchmark', icon: Database },
      { to: '/baselines', label: 'Baselines', icon: BarChart3 },
      { to: '/gate', label: 'Gate Inspector', icon: Gauge },
      { to: '/tokens', label: 'Token Accountant', icon: Coins, adminOnly: true },
    ],
  },
  {
    label: 'System',
    items: [
      { to: '/pricing', label: 'Pricing', icon: CreditCard, hostedOnly: true },
      { to: '/settings', label: 'Settings', icon: SettingsIcon, accountOnly: true },
      { to: '/admin', label: 'Admin', icon: ServerCog, adminOnly: true },
      { to: '/about', label: 'About', icon: Info },
    ],
  },
]

/** Page names for the top bar, by path. */
const TITLES: Record<string, string> = {
  '/chat': 'Chat',
  '/usage': 'Usage',
  '/run': 'Run a Task',
  '/overview': 'Overview',
  '/results': 'Results',
  '/benchmark': 'Benchmark',
  '/baselines': 'Baselines',
  '/gate': 'Gate Inspector',
  '/tokens': 'Token Accountant',
  '/pricing': 'Pricing',
  '/settings': 'Settings',
  '/admin': 'Admin',
  '/about': 'About',
  '/login': 'Log in',
  '/signup': 'Sign up',
  '/terms': 'Terms',
  '/privacy': 'Privacy',
  '/acceptable-use': 'Acceptable use',
}

/** The menu this visitor may use: operator pages are left out for non-admins when hosted. */
function useNavGroups(): { label: string; items: NavItem[] }[] {
  const { isAdmin, hosted, user } = useApp()
  const allowed = (i: NavItem) =>
    (isAdmin || !i.adminOnly) && (!i.accountOnly || (hosted && user)) && (!i.hostedOnly || hosted)
  // Hosted servers show the product page at /, so the research overview lives at /overview.
  const place = (i: NavItem) => (hosted && i.to === '/' ? { ...i, to: '/overview' } : i)
  return NAV_GROUPS.map((g) => ({ ...g, items: g.items.filter(allowed).map(place) })).filter((g) => g.items.length)
}

function BrandMark() {
  return (
    <span className="relative grid h-7 w-7 shrink-0 place-items-center rounded-md bg-navy" aria-hidden>
      <span className="h-3 w-3 rounded-full border-2 border-ice" />
      <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-ember" />
    </span>
  )
}

function Sidebar({ collapsed, onToggle, onNavigate }: { collapsed: boolean; onToggle?: () => void; onNavigate?: () => void }) {
  const groups = useNavGroups()
  const { hosted } = useApp()
  return (
    <nav aria-label="Main" className="flex h-full flex-col">
      <div className={`flex h-14 items-center border-b border-navy/10 px-3 ${collapsed ? 'justify-center' : 'justify-between'}`}>
        <Link to="/" onClick={onNavigate} className="flex items-center gap-2.5 rounded-md" aria-label="GateOrchestra home">
          <BrandMark />
          {!collapsed ? <span className="text-[15px] font-semibold tracking-tight text-ink">GateOrchestra</span> : null}
        </Link>
        {onToggle && !collapsed ? (
          <button onClick={onToggle} className="btn btn-ghost p-1.5!" aria-label="Collapse sidebar">
            <PanelLeftClose className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-3">
        {groups.map((group) => (
          <div key={group.label} className="mb-4" role="group" aria-label={group.label}>
            {collapsed ? <div className="mx-2 mb-2 h-px bg-navy/10" aria-hidden /> : <p className="eyebrow px-2.5 pb-1.5">{group.label}</p>}
            <ul className="space-y-0.5">
              {group.items.map(({ to, label, icon: Icon }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={to === '/'}
                    onClick={onNavigate}
                    title={collapsed ? label : undefined}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm transition-colors ${collapsed ? 'justify-center' : ''} ${
                        isActive ? 'bg-navy/[0.07] font-medium text-ink' : 'text-navy/70 hover:bg-navy/[0.04] hover:text-ink'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        {isActive ? <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-ember" aria-hidden /> : null}
                        <Icon className={`h-4 w-4 shrink-0 ${isActive ? 'text-ink' : 'text-navy/70 group-hover:text-navy'}`} />
                        {!collapsed ? <span>{label}</span> : null}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="border-t border-navy/10 px-3 py-3">
        {onToggle && collapsed ? (
          <button onClick={onToggle} className="btn btn-ghost w-full p-1.5!" aria-label="Expand sidebar">
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        ) : (
          <p className="eyebrow">{hosted ? 'GateOrchestra' : 'AIML Capstone · 2026'}</p>
        )}
      </div>
    </nav>
  )
}

function TokenPrompt() {
  const [token, setTokenValue] = useState('')
  return (
    <form
      className="mx-auto mt-6 flex w-full max-w-6xl flex-wrap items-center gap-2 rounded-lg border border-navy/15 bg-ice px-4 py-3 text-sm"
      onSubmit={(e) => {
        e.preventDefault()
        setToken(token.trim())
        window.location.reload()
      }}
    >
      <label htmlFor="api-token" className="font-medium text-ink">This server needs its API token</label>
      <input
        id="api-token"
        type="password"
        autoComplete="off"
        value={token}
        onChange={(e) => setTokenValue(e.target.value)}
        className="min-w-0 flex-1 rounded-md border border-navy/15 bg-soft px-3 py-1.5 font-mono text-xs outline-none focus:border-navy/40"
        placeholder="GATE_API_TOKEN"
      />
      <button type="submit" disabled={!token.trim()} className="btn btn-primary py-1.5!">
        Sign in
      </button>
      <span className="w-full text-xs text-navy/70">Kept only in this browser tab.</span>
    </form>
  )
}

/** The model that answers chat turns. The benchmark model is a different thing and is shown elsewhere. */
function useChatModel(active: boolean, online: boolean): string | null {
  const [model, setModel] = useState<string | null>(null)
  useEffect(() => {
    if (!active || !online) return
    let cancelled = false
    chatApi
      .models()
      .then((m) => {
        if (cancelled) return
        const fast = m.fast?.split('/').pop()
        const answer = m.answer?.split('/').pop()
        setModel(fast && answer && fast !== answer ? `${fast} · ${answer}` : (answer ?? fast ?? null))
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
    }
  }, [active, online])
  return model
}

function AccountControls() {
  const { hosted, user, setUser } = useApp()
  if (!hosted) return null
  if (!user) {
    return (
      <div className="flex items-center gap-1.5">
        <Link to="/login" className="btn btn-ghost py-1.5!">
          <LogIn className="h-4 w-4" /> Log in
        </Link>
        <Link to="/signup" className="btn btn-primary hidden py-1.5! sm:inline-flex">Sign up</Link>
      </div>
    )
  }
  const signOut = async () => {
    await authApi.logout().catch(() => undefined)
    setUser(null)
    // A full load of the home page: no page guard can race the redirect, and nothing of the
    // signed-out person (conversations, usage) stays in memory on a shared computer.
    window.location.assign('/')
  }
  return (
    <div className="flex items-center gap-2">
      <span className="hidden max-w-[14rem] truncate text-xs text-navy/70 lg:inline" title={user.email}>{user.email}</span>
      <button onClick={signOut} className="btn btn-secondary py-1.5!" aria-label="Log out">
        <LogOut className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Log out</span>
      </button>
    </div>
  )
}

function Banner({ tone, children }: { tone: 'warn' | 'error'; children: ReactNode }) {
  const style = tone === 'warn' ? 'border-tangerine/40 bg-tangerine/10 text-ink' : 'border-wine/30 bg-wine/5 text-wine'
  return (
    <div className={`mx-auto mt-6 flex w-full max-w-6xl flex-wrap items-center gap-2 rounded-lg border px-4 py-2.5 text-sm ${style}`} role={tone === 'error' ? 'alert' : 'status'}>
      {children}
    </div>
  )
}

function VerifyBanner() {
  const { hosted, user } = useApp()
  const [state, setState] = useState<'idle' | 'sent' | 'failed'>('idle')
  if (!hosted || !user || user.verified) return null
  const resend = () => authApi.resendVerification().then(() => setState('sent'), () => setState('failed'))
  return (
    <Banner tone="warn">
      <span>Confirm <strong className="font-semibold">{user.email}</strong> with the link we emailed you before you start chatting.</span>
      {state === 'sent' ? <span className="text-navy/70">A new link is on its way.</span> : (
        <button onClick={resend} className="btn btn-secondary ml-auto py-1!">Send the link again</button>
      )}
      {state === 'failed' ? <span className="text-wine">Could not send it; try again shortly.</span> : null}
    </Banner>
  )
}

function LegalFooter() {
  const { hosted } = useApp()
  if (!hosted) return null
  return (
    <footer className="border-t border-navy/10 px-4 md:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-5 gap-y-1 py-5 text-xs text-navy/70">
        <span>© GateOrchestra</span>
        <Link to="/terms" className="hover:text-ink">Terms</Link>
        <Link to="/privacy" className="hover:text-ink">Privacy</Link>
        <Link to="/acceptable-use" className="hover:text-ink">Acceptable use</Link>
        <Link to="/pricing" className="hover:text-ink">Pricing</Link>
        <Link to="/about" className="hover:text-ink">About</Link>
      </div>
    </footer>
  )
}

function StatusDot() {
  const { backend } = useApp()
  const label = backend === 'online' ? 'Backend online' : backend === 'error' ? 'Backend unreachable' : backend === 'locked' ? 'Sign in needed' : 'Connecting'
  const dot = backend === 'online' ? 'bg-cyan ring-2 ring-cyan/25' : backend === 'error' ? 'bg-wine' : 'bg-tangerine'
  return (
    <span className="hidden items-center gap-1.5 text-xs text-navy/70 md:inline-flex" role="status">
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  )
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const { backend, models, performanceMode, setPerformanceMode } = useApp()
  const { pathname } = useLocation()
  const onChat = pathname.startsWith('/chat')
  const chatModel = useChatModel(onChat, backend === 'online')
  const title = TITLES[pathname] ?? (pathname === '/' ? 'Home' : '')
  const has3D = pathname === '/run'
  return (
    <header className="sticky top-0 z-30 border-b border-navy/10 bg-soft/85 backdrop-blur-md">
      <div className="flex h-14 items-center gap-3 px-4 md:px-8">
        <button onClick={onMenu} className="btn btn-ghost p-1.5! md:hidden" aria-label="Open menu">
          <Menu className="h-5 w-5" />
        </button>
        <span className="text-sm font-medium text-ink">{title}</span>
        <span className="hidden h-4 w-px bg-navy/15 sm:block" aria-hidden />
        <span className="hidden min-w-0 items-center gap-2 text-xs text-navy/70 sm:flex">
          <span>{onChat ? 'Chat models' : 'Benchmark model'}</span>
          <span className="chip truncate" title={onChat ? 'The smaller model answers directly; the larger one writes answers after research' : 'The model used in the recorded benchmark'}>
            {onChat ? (chatModel ?? '—') : models ? `${models.default_model} · ${models.current_provider}` : '—'}
          </span>
        </span>
        <div className="ml-auto flex items-center gap-3">
          <StatusDot />
          {has3D ? (
            <button
              onClick={() => setPerformanceMode(!performanceMode)}
              aria-pressed={!performanceMode}
              className={`btn py-1.5! ${performanceMode ? 'btn-secondary' : 'btn-primary'}`}
              title="Show the pipeline as a 3D scene instead of the flat diagram"
            >
              <Box className="h-3.5 w-3.5" /> <span className="hidden sm:inline">3D view</span>
            </button>
          ) : null}
          <AccountControls />
        </div>
      </div>
    </header>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { backend, backendError } = useApp()

  return (
    <div className="flex min-h-screen bg-soft">
      <aside className={`sticky top-0 hidden h-screen shrink-0 border-r border-navy/10 bg-ice transition-[width] duration-200 md:block ${collapsed ? 'w-[60px]' : 'w-60'}`}>
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      </aside>

      <AnimatePresence>
        {mobileOpen ? (
          <motion.div className="fixed inset-0 z-50 md:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-ink/40" onClick={() => setMobileOpen(false)} />
            <motion.aside
              className="relative h-full w-64 border-r border-navy/10 bg-ice"
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ type: 'tween', duration: 0.18 }}
            >
              <button onClick={() => setMobileOpen(false)} className="btn btn-ghost absolute right-2 top-3 z-10 p-1.5!" aria-label="Close menu">
                <X className="h-4 w-4" />
              </button>
              <Sidebar collapsed={false} onNavigate={() => setMobileOpen(false)} />
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setMobileOpen(true)} />
        <div className="px-4 md:px-8">
          {backend === 'locked' ? <TokenPrompt /> : null}
          <VerifyBanner />
          {backend === 'error' && backendError ? <Banner tone="error">{backendError}</Banner> : null}
        </div>
        <main className="flex-1 px-4 py-8 md:px-8 md:py-10">{children}</main>
        <LegalFooter />
      </div>
    </div>
  )
}

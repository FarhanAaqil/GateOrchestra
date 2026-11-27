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
  User as UserIcon,
  X,
} from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { authApi } from '../lib/account'
import { setToken } from '../lib/auth'
import { chatApi } from '../lib/chat'
import { useApp } from '../lib/context'
import { BrandMark, Wordmark } from './BrandMark'

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

/** The signed-in person, or the build, resting at the foot of the teal sidebar. */
function SidebarFooter({ collapsed }: { collapsed: boolean }) {
  const { hosted, user, backend } = useApp()
  if (collapsed) return null
  if (!user) {
    return <p className="px-1 font-mono text-[10px] uppercase tracking-[0.1em] text-white/75">{hosted ? 'GateOrchestra' : 'AIML Capstone · 2026'}</p>
  }
  return (
    <div className="flex items-center gap-2.5 rounded-xl bg-white/10 px-3 py-2.5">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-white/15 text-white" aria-hidden>
        <UserIcon className="h-4 w-4" />
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[13px] font-medium text-white" title={user.email}>
          {user.email.split('@')[0]}
        </span>
        <span className="flex items-center gap-1.5 text-[11px] text-sage">
          <span className={`h-1.5 w-1.5 rounded-full ${backend === 'online' ? 'bg-moss' : 'bg-saffron'}`} aria-hidden />
          {backend === 'online' ? 'Online' : 'Connecting'}
        </span>
      </span>
    </div>
  )
}

function Sidebar({ collapsed, onToggle, onNavigate }: { collapsed: boolean; onToggle?: () => void; onNavigate?: () => void }) {
  const groups = useNavGroups()
  return (
    <nav aria-label="Main" className="flex h-full flex-col bg-forest text-white">
      <div className={`flex h-16 items-center px-4 ${collapsed ? 'justify-center' : 'justify-between'}`}>
        <Link to="/" onClick={onNavigate} className="flex items-center gap-2.5 rounded-md text-white" aria-label="GateOrchestra home">
          <BrandMark />
          {!collapsed ? <Wordmark /> : null}
        </Link>
        {onToggle && !collapsed ? (
          <button onClick={onToggle} className="rounded-lg p-1.5 text-white/70 transition-colors hover:bg-white/10 hover:text-white" aria-label="Collapse sidebar">
            <PanelLeftClose className="h-4 w-4" />
          </button>
        ) : null}
      </div>

      <div className="flex-1 overflow-y-auto px-3 py-2">
        {groups.map((group) => (
          <div key={group.label} className="mb-5" role="group" aria-label={group.label}>
            {collapsed ? (
              <div className="mx-2 mb-2 h-px bg-white/15" aria-hidden />
            ) : (
              <p className="px-2 pb-2 font-mono text-[10px] uppercase tracking-[0.12em] text-white/75">{group.label}</p>
            )}
            <ul className="space-y-1">
              {group.items.map(({ to, label, icon: Icon }) => (
                <li key={to}>
                  <NavLink
                    to={to}
                    end={to === '/'}
                    onClick={onNavigate}
                    title={collapsed ? label : undefined}
                    className={({ isActive }) =>
                      `group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors ${collapsed ? 'justify-center' : ''} ${
                        isActive ? 'bg-white font-semibold text-forest' : 'text-white/80 hover:bg-white/10 hover:text-white'
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon className={`h-[18px] w-[18px] shrink-0 ${isActive ? 'text-clay' : ''}`} />
                        {!collapsed ? <span className="truncate">{label}</span> : null}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="px-3 pb-4 pt-2">
        {onToggle && collapsed ? (
          <button onClick={onToggle} className="w-full rounded-lg p-1.5 text-white/70 transition-colors hover:bg-white/10 hover:text-white" aria-label="Expand sidebar">
            <PanelLeftOpen className="mx-auto h-4 w-4" />
          </button>
        ) : (
          <SidebarFooter collapsed={collapsed} />
        )}
      </div>
    </nav>
  )
}

function TokenPrompt() {
  const [token, setTokenValue] = useState('')
  return (
    <form
      className="mx-auto mt-6 flex w-full max-w-6xl flex-wrap items-center gap-2 rounded-xl border border-line bg-paper px-4 py-3 text-sm"
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
        className="min-w-0 flex-1 rounded-lg border border-line bg-cream px-3 py-1.5 font-mono text-xs outline-none focus:border-forest/40"
        placeholder="GATE_API_TOKEN"
      />
      <button type="submit" disabled={!token.trim()} className="btn btn-primary py-1.5!">
        Sign in
      </button>
      <span className="w-full text-xs text-muted">Kept only in this browser tab.</span>
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
        <Link to="/signup" className="btn btn-accent hidden py-1.5! sm:inline-flex">Sign up</Link>
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
      <span className="hidden max-w-[14rem] truncate text-xs text-muted lg:inline" title={user.email}>{user.email}</span>
      <button onClick={signOut} className="btn btn-secondary py-1.5!" aria-label="Log out">
        <LogOut className="h-3.5 w-3.5" /> <span className="hidden sm:inline">Log out</span>
      </button>
    </div>
  )
}

function Banner({ tone, children }: { tone: 'warn' | 'error'; children: ReactNode }) {
  const style = tone === 'warn' ? 'border-amber/40 bg-amber/10 text-ink' : 'border-wine/30 bg-wine/5 text-wine'
  return (
    <div className={`mx-auto mt-6 flex w-full max-w-6xl flex-wrap items-center gap-2 rounded-xl border px-4 py-2.5 text-sm ${style}`} role={tone === 'error' ? 'alert' : 'status'}>
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
      {state === 'sent' ? <span className="text-muted">A new link is on its way.</span> : (
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
    <footer className="border-t border-line px-4 md:px-8">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-5 gap-y-1 py-5 text-xs text-muted">
        <span>© GateOrchestra</span>
        <Link to="/terms" className="hover:text-forest">Terms</Link>
        <Link to="/privacy" className="hover:text-forest">Privacy</Link>
        <Link to="/acceptable-use" className="hover:text-forest">Acceptable use</Link>
        <Link to="/pricing" className="hover:text-forest">Pricing</Link>
        <Link to="/about" className="hover:text-forest">About</Link>
      </div>
    </footer>
  )
}

function StatusDot() {
  const { backend } = useApp()
  const label = backend === 'online' ? 'Backend online' : backend === 'error' ? 'Backend unreachable' : backend === 'locked' ? 'Sign in needed' : 'Connecting'
  const dot = backend === 'online' ? 'bg-moss ring-2 ring-moss/20' : backend === 'error' ? 'bg-wine' : 'bg-amber'
  return (
    <span className="hidden items-center gap-1.5 text-xs text-muted md:inline-flex" role="status">
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
    <header className="sticky top-0 z-30 border-b border-line bg-cream/90 backdrop-blur-md">
      <div className="flex h-16 items-center gap-3 px-4 md:px-8">
        <button onClick={onMenu} className="btn btn-ghost p-1.5! md:hidden" aria-label="Open menu">
          <Menu className="h-5 w-5" />
        </button>
        <span className="font-display text-[17px] font-semibold tracking-tight text-ink">{title}</span>
        <span className="hidden h-4 w-px bg-line sm:block" aria-hidden />
        <span className="hidden min-w-0 items-center gap-2 text-xs text-muted sm:flex">
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
    <div className="flex min-h-screen bg-cream">
      <aside className={`sticky top-0 hidden h-screen shrink-0 bg-forest transition-[width] duration-200 md:block ${collapsed ? 'w-[68px]' : 'w-[244px]'}`}>
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      </aside>

      <AnimatePresence>
        {mobileOpen ? (
          <motion.div className="fixed inset-0 z-50 md:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-ink/40" onClick={() => setMobileOpen(false)} />
            <motion.aside
              className="relative h-full w-[260px] bg-forest"
              initial={{ x: -270 }}
              animate={{ x: 0 }}
              exit={{ x: -270 }}
              transition={{ type: 'tween', duration: 0.18 }}
            >
              <button onClick={() => setMobileOpen(false)} className="absolute right-3 top-4 z-10 rounded-lg p-1.5 text-white/70 hover:bg-white/10 hover:text-white" aria-label="Close menu">
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

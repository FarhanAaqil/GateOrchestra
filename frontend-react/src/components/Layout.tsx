import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity,
  BarChart3,
  Coins,
  Database,
  FlaskConical,
  Gauge,
  Info,
  LayoutDashboard,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Zap,
} from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { NavLink } from 'react-router-dom'
import { useApp } from '../lib/context'
import Background from '../three/Background'

const NAV = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/run', label: 'Run a Task', icon: Play },
  { to: '/gate', label: 'Gate Inspector', icon: Gauge },
  { to: '/benchmark', label: 'Benchmark', icon: Database },
  { to: '/baselines', label: 'Baselines', icon: BarChart3 },
  { to: '/results', label: 'Results', icon: FlaskConical },
  { to: '/tokens', label: 'Token Accountant', icon: Coins },
  { to: '/about', label: 'About', icon: Info },
]

function BrandMark() {
  return (
    <div className="relative grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-ice">
      <div className="h-4 w-4 rounded-full border-[3px] border-ember" />
      <div className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-cyan" />
    </div>
  )
}

function Sidebar({ collapsed, onToggle, onNavigate }: { collapsed: boolean; onToggle?: () => void; onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex h-full flex-col gap-1">
      <div className={`mb-6 flex items-center ${collapsed ? 'justify-center' : 'justify-between'} gap-2`}>
        <div className="flex items-center gap-2.5">
          <BrandMark />
          {!collapsed ? <span className="font-display text-lg font-bold text-ice">GateOrchestra</span> : null}
        </div>
        {onToggle && !collapsed ? (
          <button onClick={onToggle} className="rounded-lg p-1.5 text-sky/70 hover:bg-ice/10 hover:text-ice" aria-label="Collapse sidebar">
            <PanelLeftClose className="h-4 w-4" />
          </button>
        ) : null}
      </div>
      {NAV.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          onClick={onNavigate}
          title={collapsed ? label : undefined}
          className={({ isActive }) =>
            `relative flex items-center gap-3 rounded-full px-3.5 py-2.5 text-sm transition-colors ${collapsed ? 'justify-center' : ''} ${
              isActive ? 'text-soft' : 'text-sky/75 hover:bg-ice/10 hover:text-ice'
            }`
          }
        >
          {({ isActive }) => (
            <>
              {isActive ? (
                <motion.span layoutId="nav-active" className="absolute inset-0 rounded-full bg-ember glow-ember" transition={{ type: 'spring', stiffness: 400, damping: 32 }} />
              ) : null}
              <Icon className="relative h-[18px] w-[18px] shrink-0" />
              {!collapsed ? <span className="relative font-medium">{label}</span> : null}
            </>
          )}
        </NavLink>
      ))}
      {onToggle && collapsed ? (
        <button onClick={onToggle} className="mt-auto rounded-lg p-2 text-sky/70 hover:bg-ice/10 hover:text-ice" aria-label="Expand sidebar">
          <PanelLeftOpen className="mx-auto h-4 w-4" />
        </button>
      ) : (
        <p className="mt-auto px-3 font-mono text-[10px] uppercase tracking-widest text-sky/40">AIML Capstone · 2026</p>
      )}
    </nav>
  )
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const { backend, models, performanceMode, setPerformanceMode } = useApp()
  return (
    <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-sky/60 bg-ice/80 px-4 py-3 backdrop-blur-xl md:px-8">
      <button onClick={onMenu} className="rounded-lg p-2 text-navy hover:bg-soft md:hidden" aria-label="Open menu">
        <Menu className="h-5 w-5" />
      </button>
      <div className="flex min-w-0 items-center gap-2 font-mono text-xs">
        <span className="hidden text-navy/60 sm:inline">Model</span>
        <span className="truncate rounded-full bg-navy px-3 py-1 text-ice">
          {models ? models.default_model : '—'}
          <span className="text-cyan"> · {models?.current_provider ?? '…'}</span>
        </span>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-xs ${
            backend === 'online' ? 'border-sky bg-soft text-navy' : backend === 'error' ? 'border-wine bg-wine text-soft' : 'border-sky text-navy/60'
          }`}
          role="status"
        >
          <span className={`h-2 w-2 rounded-full ${backend === 'online' ? 'bg-cyan glow-cyan' : backend === 'error' ? 'bg-lemon' : 'bg-sky'}`} />
          <Activity className="h-3.5 w-3.5" aria-hidden />
          {backend === 'online' ? 'Backend online' : backend === 'error' ? 'Backend unreachable' : 'Connecting'}
        </span>
        <button
          onClick={() => setPerformanceMode(!performanceMode)}
          aria-pressed={performanceMode}
          className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 font-mono text-xs transition-colors ${
            performanceMode ? 'border-lemon bg-lemon text-ink' : 'border-sky text-navy hover:bg-soft'
          }`}
          title="Performance mode turns off the 3D scenes"
        >
          <Zap className="h-3.5 w-3.5" />
          <span className="hidden sm:inline">Performance</span>
        </button>
      </div>
    </header>
  )
}

export default function Layout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const { backend, backendError } = useApp()

  return (
    <div className="grain flex min-h-screen">
      <Background />
      <aside className={`sticky top-0 hidden h-screen shrink-0 bg-navy p-3 transition-[width] duration-300 md:block ${collapsed ? 'w-[76px]' : 'w-64'}`}>
        <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
      </aside>

      <AnimatePresence>
        {mobileOpen ? (
          <motion.div className="fixed inset-0 z-50 md:hidden" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <div className="absolute inset-0 bg-navy/50" onClick={() => setMobileOpen(false)} />
            <motion.aside
              className="relative h-full w-64 bg-navy p-3"
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              <Sidebar collapsed={false} onNavigate={() => setMobileOpen(false)} />
            </motion.aside>
          </motion.div>
        ) : null}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setMobileOpen(true)} />
        {backend === 'error' && backendError ? (
          <div className="mx-4 mt-4 rounded-2xl border border-wine/30 bg-wine/10 px-4 py-3 text-sm text-wine md:mx-8" role="alert">
            {backendError}
          </div>
        ) : null}
        <main className="flex-1 px-4 py-8 md:px-10 md:py-10">{children}</main>
      </div>
    </div>
  )
}

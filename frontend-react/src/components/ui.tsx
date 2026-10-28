import { animate, motion, useMotionValue, useReducedMotion, useSpring } from 'framer-motion'
import { AlertTriangle, CheckCircle2, CircleDashed, Info } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { rise, stagger } from '../lib/motion'
import type { ResultsStatus } from '../lib/types'

/** Card that tilts slightly toward the pointer. `navy` is the dark control-room panel. */
export function Card({
  children,
  className = '',
  tilt = true,
  navy = false,
}: {
  children: ReactNode
  className?: string
  tilt?: boolean
  navy?: boolean
}) {
  const reduce = useReducedMotion()
  const rx = useSpring(useMotionValue(0), { stiffness: 200, damping: 20 })
  const ry = useSpring(useMotionValue(0), { stiffness: 200, damping: 20 })
  const enabled = tilt && !reduce

  return (
    <motion.div
      variants={rise}
      className={`${navy ? 'panel' : 'card'} p-5 md:p-6 ${className}`}
      style={enabled ? { rotateX: rx, rotateY: ry, transformPerspective: 1000 } : undefined}
      onPointerMove={(e) => {
        if (!enabled) return
        const r = e.currentTarget.getBoundingClientRect()
        ry.set(((e.clientX - r.left) / r.width - 0.5) * 5)
        rx.set(-((e.clientY - r.top) / r.height - 0.5) * 5)
      }}
      onPointerLeave={() => {
        rx.set(0)
        ry.set(0)
      }}
    >
      {children}
    </motion.div>
  )
}

/** Animated number that counts up to its value. */
export function CountUp({
  value,
  decimals = 0,
  suffix = '',
  className = '',
}: {
  value: number
  decimals?: number
  suffix?: string
  className?: string
}) {
  const reduce = useReducedMotion()
  const [shown, setShown] = useState(reduce ? value : 0)
  const from = useRef(0)

  useEffect(() => {
    if (reduce) {
      from.current = value
      return
    }
    const controls = animate(from.current, value, { duration: 1.1, ease: 'easeOut', onUpdate: setShown })
    from.current = value
    return () => controls.stop()
  }, [value, reduce])

  return (
    <span className={`font-mono tabular-nums ${className}`}>
      {(reduce ? value : shown).toFixed(decimals)}
      {suffix}
    </span>
  )
}

/** Big numbered page header, after the "09 / Ice & Ember" swatch cards. */
export function PageHeader({
  number,
  kicker,
  title,
  children,
}: {
  number: string
  kicker: string
  title: ReactNode
  children?: ReactNode
}) {
  return (
    <motion.header variants={rise} className="mb-8">
      <p className="font-mono text-sm text-navy/60">
        <span className="text-ember">{number}</span> — {kicker}
      </p>
      <h1 className="mt-2 text-4xl font-bold leading-[1.05] text-ink md:text-5xl">{title}</h1>
      {children ? <div className="mt-3 max-w-3xl text-base text-navy/70">{children}</div> : null}
    </motion.header>
  )
}

/** Wraps a page so its cards enter with a stagger. */
export function Page({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, y: -8, transition: { duration: 0.2 } }}
      variants={stagger}
      className="mx-auto w-full max-w-7xl"
    >
      {children}
    </motion.div>
  )
}

export function Tag({ children, tone = 'sky' }: { children: ReactNode; tone?: 'sky' | 'lemon' | 'ember' | 'navy' | 'wine' }) {
  const tones = {
    sky: 'border-sky bg-ice text-navy',
    lemon: 'border-lemon bg-lemon text-ink',
    ember: 'border-ember bg-ember text-soft',
    navy: 'border-navy bg-navy text-ice',
    wine: 'border-wine bg-wine text-soft',
  }
  return <span className={`inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-medium ${tones[tone]}`}>{children}</span>
}

export function PillLink({ to, children, variant = 'ember' }: { to: string; children: ReactNode; variant?: 'ember' | 'navy' | 'ghost' }) {
  const styles = {
    ember: 'bg-ember text-soft glow-ember hover:-translate-y-0.5',
    navy: 'bg-navy text-ice hover:-translate-y-0.5',
    ghost: 'border border-navy/20 text-navy hover:bg-soft',
  }
  return (
    <Link to={to} className={`inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-semibold transition-all ${styles[variant]}`}>
      {children}
    </Link>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <motion.div variants={rise} className="card flex flex-col items-center gap-3 border-dashed px-6 py-12 text-center">
      <CircleDashed className="h-8 w-8 text-navy/40" aria-hidden />
      <h3 className="text-lg font-semibold text-ink">{title}</h3>
      {children ? <div className="max-w-xl text-sm text-navy/70">{children}</div> : null}
    </motion.div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-2xl border border-wine/30 bg-wine/10 px-4 py-3 text-sm text-wine" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{message}</span>
    </div>
  )
}

const STATUS_STYLE: Record<ResultsStatus, { label: string; cls: string; icon: ReactNode }> = {
  verified: { label: 'Verified', cls: 'border-navy bg-navy text-cyan', icon: <CheckCircle2 className="h-3.5 w-3.5" /> },
  partial: { label: 'Smoke run', cls: 'border-tangerine bg-tangerine/15 text-ink', icon: <Info className="h-3.5 w-3.5" /> },
  unverified: { label: 'Unverified', cls: 'border-wine bg-wine text-soft', icon: <AlertTriangle className="h-3.5 w-3.5" /> },
  no_results: { label: 'No results', cls: 'border-navy/20 bg-soft text-navy', icon: <CircleDashed className="h-3.5 w-3.5" /> },
}

export function StatusBadge({ status }: { status: ResultsStatus }) {
  const s = STATUS_STYLE[status]
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 font-mono text-[11px] uppercase tracking-wider ${s.cls}`}>
      {s.icon}
      {s.label}
    </span>
  )
}

/** Explains why recorded results are not shown, in place of a chart. */
export function NoCitableResults({ status, what }: { status: ResultsStatus | undefined; what: string }) {
  return (
    <EmptyState title={`No verified ${what} yet`}>
      {status === 'unverified' ? (
        <p>
          The only recorded run is a 5-task, 1-seed smoke run that fails{' '}
          <code className="rounded bg-lemon px-1 font-mono text-ink">verify_final_results.py</code>, so it is not shown as a finding.
        </p>
      ) : (
        <p>No benchmark results have been recorded.</p>
      )}
      <p className="mt-2">
        Run <code className="rounded bg-lemon px-1 font-mono text-ink">python scripts/final_benchmark.py</code> to fill this page
        with real numbers.
      </p>
    </EmptyState>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-navy/70" role="status">
      <span className="h-3 w-3 animate-spin rounded-full border-2 border-ember border-t-transparent" />
      {label}
    </div>
  )
}

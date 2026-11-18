import { animate, motion, useReducedMotion } from 'framer-motion'
import { AlertTriangle, CheckCircle2, CircleDashed, Info } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { GLOSSARY, type GlossaryTerm } from '../lib/glossary'
import { rise, stagger } from '../lib/motion'
import type { ResultsStatus } from '../lib/types'

/** A technical term with its plain-language meaning on hover or keyboard focus. */
export function Term({ term, children }: { term: GlossaryTerm; children: ReactNode }) {
  return (
    <abbr title={GLOSSARY[term]} tabIndex={0} className="cursor-help no-underline decoration-navy/30 decoration-dotted underline-offset-4 hover:underline">
      {children}
    </abbr>
  )
}

/**
 * A panel. `navy` is the dark instrument panel used where numbers should stand out.
 * `tilt` is accepted for older call sites and ignored: panels no longer move.
 */
export function Card({
  children,
  className = '',
  navy = false,
}: {
  children: ReactNode
  className?: string
  tilt?: boolean
  navy?: boolean
}) {
  return (
    <motion.div variants={rise} className={`${navy ? 'panel' : 'card'} p-5 ${className}`}>
      {children}
    </motion.div>
  )
}

/** A number that eases up to its value once (instant with reduced motion). */
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
    const controls = animate(from.current, value, { duration: 0.7, ease: 'easeOut', onUpdate: setShown })
    from.current = value
    return () => controls.stop()
  }, [value, reduce])

  return (
    <span className={`num ${className}`}>
      {(reduce ? value : shown).toFixed(decimals)}
      {suffix}
    </span>
  )
}

/** Page title block: a small section label, the title, and one or two lines of context. */
export function PageHeader({
  kicker,
  title,
  children,
  actions,
}: {
  /** Kept for older call sites; page numbers are no longer shown. */
  number?: string
  kicker: string
  title: ReactNode
  children?: ReactNode
  /** Buttons aligned to the right of the title. */
  actions?: ReactNode
}) {
  return (
    <motion.header variants={rise} className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-navy/10 pb-6">
      <div className="min-w-0 max-w-3xl">
        <p className="eyebrow">{kicker}</p>
        <h1 className="mt-2 text-[28px] font-semibold leading-tight tracking-tight text-ink md:text-[32px]">{title}</h1>
        {children ? <div className="mt-2 text-[15px] leading-relaxed text-navy/70">{children}</div> : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
    </motion.header>
  )
}

/** A section heading inside a page. */
export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <motion.div variants={rise} className="mb-3 mt-10 flex flex-wrap items-center gap-3">
      <h2 className="text-lg font-semibold text-ink">{children}</h2>
      {aside}
    </motion.div>
  )
}

/** Wraps a page so its panels fade in together. */
export function Page({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial="hidden"
      animate="show"
      exit={{ opacity: 0, transition: { duration: 0.12 } }}
      variants={stagger}
      className="mx-auto w-full max-w-6xl"
    >
      {children}
    </motion.div>
  )
}

export function Tag({ children, tone = 'sky' }: { children: ReactNode; tone?: 'sky' | 'lemon' | 'ember' | 'navy' | 'wine' }) {
  const tones = {
    sky: '',
    lemon: 'border-tangerine/40! bg-tangerine/10!',
    ember: 'border-ember/40! bg-ember/10! text-ink!',
    navy: 'border-ice/15! bg-ice/5! text-sky!',
    wine: 'border-wine/30! bg-wine/5! text-wine!',
  }
  return <span className={`chip ${tones[tone]}`}>{children}</span>
}

export function PillLink({ to, children, variant = 'ember' }: { to: string; children: ReactNode; variant?: 'ember' | 'navy' | 'ghost' }) {
  const styles = { ember: 'btn-accent', navy: 'btn-primary', ghost: 'btn-secondary' }
  return (
    <Link to={to} className={`btn ${styles[variant]}`}>
      {children}
    </Link>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <motion.div variants={rise} className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-navy/20 px-6 py-12 text-center">
      <CircleDashed className="h-6 w-6 text-navy/35" aria-hidden />
      <h3 className="text-[15px] font-semibold text-ink">{title}</h3>
      {children ? <div className="max-w-xl text-sm text-navy/65">{children}</div> : null}
    </motion.div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-wine/25 bg-wine/5 px-3.5 py-2.5 text-sm text-wine" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{message}</span>
    </div>
  )
}

const STATUS_STYLE: Record<ResultsStatus, { label: string; cls: string; icon: ReactNode }> = {
  verified: { label: 'Verified', cls: 'border-navy/15 text-navy', icon: <CheckCircle2 className="h-3 w-3 text-navy" /> },
  partial: { label: 'Partial run', cls: 'border-tangerine/50 bg-tangerine/10 text-ink', icon: <Info className="h-3 w-3" /> },
  unverified: { label: 'Unverified', cls: 'border-wine/40 bg-wine/5 text-wine', icon: <AlertTriangle className="h-3 w-3" /> },
  no_results: { label: 'No results', cls: 'border-navy/15 text-navy/70', icon: <CircleDashed className="h-3 w-3" /> },
}

export function StatusBadge({ status }: { status: ResultsStatus }) {
  const s = STATUS_STYLE[status]
  return (
    <span className={`chip ${s.cls}`}>
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
          The recorded results fail <code className="rounded bg-navy/5 px-1 font-mono text-ink">verify_final_results.py</code>, so they are
          not shown as findings.
        </p>
      ) : (
        <p>No benchmark results have been recorded.</p>
      )}
      <p className="mt-2">
        Run <code className="rounded bg-navy/5 px-1 font-mono text-ink">python scripts/final_benchmark.py</code> to fill this page with
        real numbers.
      </p>
    </EmptyState>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-navy/65" role="status">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-navy/20 border-t-navy" />
      {label}
    </div>
  )
}

/** A placeholder block shaped like the content that is loading. */
export function Skeleton({ className = '' }: { className?: string }) {
  return <span aria-hidden className={`block animate-pulse rounded-md bg-navy/[0.07] motion-reduce:animate-none ${className}`} />
}

/** Loading placeholder for a grid of panels (stats, charts). */
export function SkeletonCards({ count = 3, height = 'h-28' }: { count?: number; height?: string }) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3" role="status" aria-label="Loading">
      {Array.from({ length: count }, (_, i) => (
        <div key={i} className="card space-y-3 p-5">
          <Skeleton className="h-3 w-1/3" />
          <Skeleton className={height} />
        </div>
      ))}
    </div>
  )
}

/** Loading placeholder for a table. */
export function SkeletonTable({ rows = 6 }: { rows?: number }) {
  return (
    <div className="card space-y-3 p-5" role="status" aria-label="Loading">
      <Skeleton className="h-5 w-full" />
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-4 w-full" />
      ))}
    </div>
  )
}

/** One number with its label, for stat rows. */
export function Stat({ label, value, note, tone = 'ink' }: { label: string; value: ReactNode; note?: ReactNode; tone?: 'ink' | 'ember' | 'wine' }) {
  const colour = { ink: 'text-ink', ember: 'text-ember', wine: 'text-wine' }[tone]
  return (
    <Card>
      <p className="eyebrow">{label}</p>
      <p className={`num mt-2 text-[28px] font-medium leading-none ${colour}`}>{value}</p>
      {note ? <p className="mt-2 text-xs text-navy/60">{note}</p> : null}
    </Card>
  )
}

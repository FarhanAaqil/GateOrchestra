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
    <abbr title={GLOSSARY[term]} tabIndex={0} className="cursor-help no-underline decoration-clay/40 decoration-dotted underline-offset-4 hover:underline">
      {children}
    </abbr>
  )
}

/**
 * A panel. `forest` is the teal instrument panel used where numbers should lead.
 * `tilt` is accepted for older call sites and ignored: panels no longer move.
 */
export function Card({
  children,
  className = '',
  forest = false,
}: {
  children: ReactNode
  className?: string
  tilt?: boolean
  forest?: boolean
}) {
  return (
    <motion.div variants={rise} className={`${forest ? 'panel' : 'card'} p-5 ${className}`}>
      {children}
    </motion.div>
  )
}

/**
 * A card with a filled burgundy header strip, for the panels that lead a page.
 * `aside` sits at the right of the strip (a live dot, a count, a link).
 */
export function HeadedCard({
  title,
  icon,
  aside,
  children,
  className = '',
  bodyClassName = 'p-5',
}: {
  title: ReactNode
  icon?: ReactNode
  aside?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <motion.section variants={rise} className={`card overflow-hidden p-0! ${className}`}>
      <header className="card-head">
        {icon ? <span className="shrink-0 text-clay-light" aria-hidden>{icon}</span> : null}
        <h2 className="min-w-0 truncate text-[17px] font-semibold text-white">{title}</h2>
        {aside ? <div className="ml-auto flex items-center gap-2 text-xs text-clay-light">{aside}</div> : null}
      </header>
      <div className={bodyClassName}>{children}</div>
    </motion.section>
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

/** Page title block: a small section label, the serif title, and a line of context. */
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
    <motion.header variants={rise} className="mb-8 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-6">
      <div className="min-w-0 max-w-3xl">
        <p className="eyebrow">{kicker}</p>
        <h1 className="mt-2.5 font-display text-[30px] font-semibold leading-[1.15] tracking-tight text-forest md:text-[36px]">{title}</h1>
        <span className="mt-3 block h-px w-14 bg-clay" aria-hidden />
        {children ? <div className="mt-3 text-[15px] leading-relaxed text-muted">{children}</div> : null}
      </div>
      {actions ? <div className="flex flex-wrap gap-2">{actions}</div> : null}
    </motion.header>
  )
}

/** A section heading inside a page. */
export function SectionTitle({ children, aside }: { children: ReactNode; aside?: ReactNode }) {
  return (
    <motion.div variants={rise} className="mb-3 mt-10 flex flex-wrap items-center gap-3">
      <h2 className="font-display text-[20px] font-semibold tracking-tight text-ink">{children}</h2>
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

export function Tag({ children, tone = 'sage' }: { children: ReactNode; tone?: 'sage' | 'butter' | 'clay' | 'forest' | 'wine' }) {
  const tones = {
    sage: '',
    butter: 'border-amber/35! bg-amber/10! text-ink!',
    clay: 'border-clay/35! bg-clay/8! text-clay!',
    forest: 'border-white/20! bg-white/10! text-sage!',
    wine: 'border-wine/30! bg-wine/6! text-wine!',
  }
  return <span className={`chip ${tones[tone]}`}>{children}</span>
}

export function PillLink({ to, children, variant = 'clay' }: { to: string; children: ReactNode; variant?: 'clay' | 'forest' | 'ghost' }) {
  const styles = { clay: 'btn-accent', forest: 'btn-primary', ghost: 'btn-secondary' }
  return (
    <Link to={to} className={`btn ${styles[variant]}`}>
      {children}
    </Link>
  )
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <motion.div variants={rise} className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-line bg-tile/60 px-6 py-12 text-center">
      <CircleDashed className="h-6 w-6 text-forest" aria-hidden />
      <h3 className="font-display text-[17px] font-semibold text-ink">{title}</h3>
      {children ? <div className="max-w-xl text-sm text-muted">{children}</div> : null}
    </motion.div>
  )
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-xl border border-wine/25 bg-wine/5 px-3.5 py-2.5 text-sm text-wine" role="alert">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{message}</span>
    </div>
  )
}

const STATUS_STYLE: Record<ResultsStatus, { label: string; cls: string; icon: ReactNode }> = {
  verified: { label: 'Verified', cls: 'border-moss/35 bg-moss/8 text-moss', icon: <CheckCircle2 className="h-3 w-3 text-moss" /> },
  partial: { label: 'Partial run', cls: 'border-amber/40 bg-amber/10 text-ink', icon: <Info className="h-3 w-3" /> },
  unverified: { label: 'Unverified', cls: 'border-wine/35 bg-wine/5 text-wine', icon: <AlertTriangle className="h-3 w-3" /> },
  no_results: { label: 'No results', cls: 'border-line text-muted', icon: <CircleDashed className="h-3 w-3" /> },
}

/** `onDark` is the variant that sits on a burgundy or teal strip, where the tinted one would not read. */
export function StatusBadge({ status, onDark = false }: { status: ResultsStatus; onDark?: boolean }) {
  const s = STATUS_STYLE[status]
  if (onDark) {
    return (
      <span className="chip border-white/35! bg-white/15! text-white!">
        {s.label}
      </span>
    )
  }
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
          The recorded results fail <code className="rounded bg-tile px-1 font-mono text-ink">verify_final_results.py</code>, so they are
          not shown as findings.
        </p>
      ) : (
        <p>No benchmark results have been recorded.</p>
      )}
      <p className="mt-2">
        Run <code className="rounded bg-tile px-1 font-mono text-ink">python scripts/final_benchmark.py</code> to fill this page with
        real numbers.
      </p>
    </EmptyState>
  )
}

export function Spinner({ label = 'Loading' }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted" role="status">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-line border-t-forest" />
      {label}
    </div>
  )
}

/** A placeholder block shaped like the content that is loading. */
export function Skeleton({ className = '' }: { className?: string }) {
  return <span aria-hidden className={`block animate-pulse rounded-md bg-tile motion-reduce:animate-none ${className}`} />
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

/**
 * One number with its label. `clay` marks the stat that matters with a burgundy
 * rule down its left edge, rather than tinting the number itself.
 */
export function Stat({ label, value, note, tone = 'ink' }: { label: string; value: ReactNode; note?: ReactNode; tone?: 'ink' | 'clay' | 'wine' }) {
  const colour = { ink: 'text-ink', clay: 'text-ink', wine: 'text-wine' }[tone]
  return (
    <Card className={tone === 'clay' ? 'border-l-[3px]! border-l-clay!' : ''}>
      <p className="eyebrow">{label}</p>
      <p className={`num mt-2.5 text-[30px] font-semibold leading-none ${colour}`}>{value}</p>
      {note ? <p className="mt-2.5 text-xs leading-relaxed text-muted">{note}</p> : null}
    </Card>
  )
}

/**
 * A compact stat inside a card: icon, label, number. Used in rows of two or
 * three where a full Stat panel would be too heavy.
 */
export function StatTile({ label, value, icon, note }: { label: string; value: ReactNode; icon?: ReactNode; note?: ReactNode }) {
  return (
    <div className="tile flex items-center gap-3 px-4 py-3.5">
      {icon ? <span className="shrink-0 text-forest" aria-hidden>{icon}</span> : null}
      <div className="min-w-0">
        <p className="truncate text-xs font-medium text-muted">{label}</p>
        <p className="num mt-0.5 text-[22px] font-semibold leading-none text-ink">{value}</p>
        {note ? <p className="mt-1 truncate text-[11px] text-muted">{note}</p> : null}
      </div>
    </div>
  )
}

import { motion, useReducedMotion } from 'framer-motion'
import { num } from '../lib/format'

interface Props {
  /** Tokens the probe actually spent (prompt + completion). */
  probeTokens: number | null
  /** Tokens the MAS actually spent, if it ran. */
  masTokens: number | null
  /** MAS budget cap for this run (k x probe tokens), if MAS was allowed. */
  cap: number | null
  size?: number
}

function Arc({ r, fraction, color, width, delay }: { r: number; fraction: number; color: string; width: number; delay: number }) {
  const reduce = useReducedMotion()
  const c = 2 * Math.PI * r
  const length = Math.min(Math.max(fraction, 0), 1) * c
  return (
    <motion.circle
      r={r}
      cx="0"
      cy="0"
      fill="none"
      stroke={color}
      strokeWidth={width}
      strokeLinecap="round"
      strokeDasharray={`${c} ${c}`}
      initial={{ strokeDashoffset: c }}
      animate={{ strokeDashoffset: c - length }}
      transition={{ duration: reduce ? 0 : 1.2, ease: 'easeOut', delay: reduce ? 0 : delay }}
      transform="rotate(-90)"
      style={{ filter: `drop-shadow(0 0 6px ${color})` }}
    />
  )
}

/**
 * Fuel-gauge rings: probe spend (inner) and MAS spend against its cap (outer).
 * The outer ring turns coral when the MAS exceeded its cap.
 */
export default function TokenGauge({ probeTokens, masTokens, cap, size = 200 }: Props) {
  const over = masTokens != null && cap != null && masTokens > cap
  const masFraction = masTokens != null && cap ? masTokens / cap : 0
  const probeFraction = probeTokens != null && cap ? probeTokens / cap : probeTokens ? 1 : 0
  const total = (probeTokens ?? 0) + (masTokens ?? 0)
  return (
    <div className="flex flex-col items-center gap-3">
      <svg width={size} height={size} viewBox="-100 -100 200 200" role="img" aria-label={`Token spend ${total}`}>
        <circle r="82" fill="none" stroke="rgba(27,38,59,0.08)" strokeWidth="10" />
        <circle r="62" fill="none" stroke="rgba(27,38,59,0.08)" strokeWidth="8" />
        <Arc r={82} fraction={masFraction} color={over ? '#8B1538' : '#FF8C42'} width={10} delay={0.35} />
        <Arc r={62} fraction={probeFraction} color="#A8DBFF" width={8} delay={0} />
        <text y="-4" textAnchor="middle" fill="#1B1B1F" className="font-mono text-[26px] font-semibold">
          {num(total)}
        </text>
        <text y="18" textAnchor="middle" fill="rgba(27,38,59,0.6)" className="font-mono text-[10px] uppercase tracking-widest">
          tokens
        </text>
      </svg>
      <dl className="grid w-full grid-cols-3 gap-2 text-center font-mono text-xs">
        <div>
          <dt className="text-navy/60">Probe</dt>
          <dd className="text-navy">{num(probeTokens)}</dd>
        </div>
        <div>
          <dt className="text-navy/60">MAS</dt>
          <dd className={over ? 'font-semibold text-wine' : 'text-tangerine'}>{num(masTokens)}</dd>
        </div>
        <div>
          <dt className="text-navy/60">MAS cap</dt>
          <dd className="text-ink">{cap == null ? '—' : num(cap)}</dd>
        </div>
      </dl>
      {over ? <p className="rounded-full bg-wine px-3 py-1 font-mono text-xs text-soft">MAS spent {num(masTokens! - cap!)} over its cap</p> : null}
    </div>
  )
}

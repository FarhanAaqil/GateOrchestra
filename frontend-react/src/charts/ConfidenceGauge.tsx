import { motion, useReducedMotion } from 'framer-motion'

/**
 * Semicircle gauge of the gate's escalation probability, with an optional
 * threshold marker. 0 = certain STOP, 1 = certain ESCALATE.
 */
export default function ConfidenceGauge({
  escalateProbability,
  threshold,
}: {
  escalateProbability: number
  threshold: number | null
}) {
  const reduce = useReducedMotion()
  const angle = (p: number) => Math.PI * (1 - p)
  const needle = angle(escalateProbability)
  const r = 80
  const tick = threshold == null ? null : angle(threshold)
  return (
    <svg viewBox="-100 -96 200 112" className="w-full max-w-xs" role="img" aria-label={`Escalation probability ${(escalateProbability * 100).toFixed(0)}%`}>
      <defs>
        <linearGradient id="cg" x1="0" x2="1">
          <stop offset="0%" stopColor="#A8DBFF" />
          <stop offset="50%" stopColor="#00E5FF" />
          <stop offset="100%" stopColor="#FF6B35" />
        </linearGradient>
      </defs>
      <path d={`M ${-r} 0 A ${r} ${r} 0 0 1 ${r} 0`} fill="none" stroke="url(#cg)" strokeWidth="12" strokeLinecap="round" opacity="0.85" />
      {tick != null ? (
        <line x1={Math.cos(tick) * (r - 14)} y1={-Math.sin(tick) * (r - 14)} x2={Math.cos(tick) * (r + 10)} y2={-Math.sin(tick) * (r + 10)} stroke="#8B1538" strokeWidth="3" />
      ) : null}
      <motion.line
        x1="0"
        y1="0"
        x2={r - 18}
        y2="0"
        stroke="#1B263B"
        strokeWidth="3"
        strokeLinecap="round"
        initial={{ rotate: -180 }}
        animate={{ rotate: (-needle * 180) / Math.PI }}
        transition={{ duration: reduce ? 0 : 1, type: 'spring', stiffness: 60, damping: 12 }}
      />
      <circle r="5" fill="#1B263B" />
      <text x={-r} y="14" textAnchor="middle" fill="#1B263B" className="font-mono text-[9px]">STOP</text>
      <text x={r} y="14" textAnchor="middle" fill="#FF6B35" className="font-mono text-[9px]">ESCALATE</text>
    </svg>
  )
}

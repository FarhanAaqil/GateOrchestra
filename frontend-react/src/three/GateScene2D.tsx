import { useEffect, useRef } from 'react'
import { STRATEGY_LABELS } from '../lib/format'
import { agentPosition, LAYOUT, orbPosition, STRATEGIES, type Vec3 } from './gateTimeline'
import { useGateRun, type GateRunInput } from './useGateRun'

// World units -> SVG pixels (viewBox 0 0 640 320).
const sx = (x: number) => 320 + x * 52
const sy = (y: number) => 150 - y * 52
const pt = (v: Vec3) => `${sx(v[0]).toFixed(1)},${sy(v[1]).toFixed(1)}`

/** Lightweight SVG version of the gate scene (mobile, performance mode, reduced motion). */
export default function GateScene2D(props: GateRunInput & { animate?: boolean }) {
  const sample = useGateRun(props)
  const orb = useRef<SVGCircleElement>(null)
  const agents = useRef<(SVGGElement | null)[]>([])
  const cot = useRef<SVGCircleElement>(null)
  const animateMotion = props.animate ?? true
  const { gate, cot: cotPos, core } = LAYOUT

  useEffect(() => {
    let frame = 0
    const start = performance.now() / 1000
    const tick = () => {
      const now = performance.now() / 1000
      const clock = animateMotion ? now - start : 0
      const run = sample(now)
      const { position, visible, atTarget } = orbPosition(run.phase, run.elapsed, run.decision, run.strategy, clock)
      if (orb.current) {
        orb.current.setAttribute('cx', String(sx(position[0])))
        orb.current.setAttribute('cy', String(sy(position[1])))
        orb.current.style.opacity = visible ? '1' : '0'
      }
      if (cot.current) {
        cot.current.style.opacity = run.phase === 'route' && run.decision === 'STOP' && atTarget ? '1' : '0.45'
      }
      STRATEGIES.forEach((strategy, i) => {
        const p = agentPosition(i, clock)
        const g = agents.current[i]
        if (!g) return
        g.setAttribute('transform', `translate(${sx(p[0])},${sy(p[1])})`)
        const chosen = run.phase === 'route' && run.decision === 'ESCALATE' && run.strategy === strategy
        g.style.opacity = chosen ? '1' : '0.45'
      })
      frame = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(frame)
  }, [sample, animateMotion])

  return (
    <svg viewBox="0 0 640 320" className="h-full w-full" role="img" aria-label="Gate routing diagram">
      <defs>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <line x1={sx(LAYOUT.start[0])} y1={sy(LAYOUT.start[1])} x2={sx(gate[0])} y2={sy(gate[1])} stroke="#00E5FF" strokeOpacity="0.45" strokeDasharray="5 5" />
      <path d={`M${pt(gate)} Q${sx(gate[0] + 1.2)},${sy((gate[1] + cotPos[1]) / 2)} ${pt(cotPos)}`} stroke="#A8DBFF" strokeOpacity="0.8" fill="none" strokeWidth="2" />
      <path d={`M${pt(gate)} Q${sx((gate[0] + core[0]) / 2)},${sy(gate[1] + 1.1)} ${pt(core)}`} stroke="#FF6B35" strokeOpacity="0.8" fill="none" strokeWidth="2" />

      <g filter="url(#glow)">
        <circle cx={sx(gate[0])} cy={sy(gate[1])} r="46" fill="none" stroke="#00E5FF" strokeWidth="3" />
        <circle cx={sx(gate[0])} cy={sy(gate[1])} r="34" fill="#00E5FF" fillOpacity="0.08" stroke="#00E5FF" strokeOpacity="0.5" />
        <circle ref={cot} cx={sx(cotPos[0])} cy={sy(cotPos[1])} r="16" fill="#A8DBFF" style={{ transition: 'opacity .3s' }} />
        <polygon
          points={[0, 1, 2, 3, 4, 5].map((i) => `${sx(core[0]) + 22 * Math.cos((i * Math.PI) / 3)},${sy(core[1]) + 22 * Math.sin((i * Math.PI) / 3)}`).join(' ')}
          fill="#FF6B35"
          fillOpacity="0.9"
        />
        {STRATEGIES.map((strategy, i) => (
          <g key={strategy} ref={(el) => {
            agents.current[i] = el
          }} style={{ transition: 'opacity .3s' }}>
            <circle r="9" fill="#FFF56B" />
            <text y="-14" textAnchor="middle" fill="#FFF56B" className="font-mono text-[11px]">
              {STRATEGY_LABELS[strategy]}
            </text>
          </g>
        ))}
        <circle ref={orb} r="8" fill="#F7F7F7" style={{ opacity: 0 }} />
      </g>
      <text x={sx(gate[0])} y={sy(gate[1]) - 58} textAnchor="middle" fill="#00E5FF" className="font-mono text-[12px]">Gate</text>
      <text x={sx(cotPos[0])} y={sy(cotPos[1]) + 32} textAnchor="middle" fill="#A8DBFF" className="font-mono text-[12px]">CoT-SC answer</text>
      <text x={sx(core[0])} y={sy(core[1] - 1.62)} textAnchor="middle" fill="#FF6B35" className="font-mono text-[12px]">Orchestrator</text>
    </svg>
  )
}

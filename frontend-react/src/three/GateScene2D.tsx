import { useEffect, useRef } from 'react'
import { STRATEGY_LABELS } from '../lib/format'
import { agentPosition, LAYOUT, orbPosition, STRATEGIES, type Vec3 } from './gateTimeline'
import { useGateRun, type GateRunInput } from './useGateRun'

// World units -> SVG pixels (viewBox 0 0 640 320).
const sx = (x: number) => 320 + x * 52
const sy = (y: number) => 150 - y * 52
const pt = (v: Vec3) => `${sx(v[0]).toFixed(1)},${sy(v[1]).toFixed(1)}`

/** Flat SVG view of the gate scene: the default, and always used on phones and with reduced motion. */
export default function GateScene2D(props: GateRunInput & { animate?: boolean }) {
  const sample = useGateRun(props)
  const orb = useRef<SVGCircleElement>(null)
  const agents = useRef<(SVGCircleElement | null)[]>([])
  const agentLabels = useRef<(SVGTextElement | null)[]>([])
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
        const dot = agents.current[i]
        const label = agentLabels.current[i]
        if (!dot || !label) return
        // Plain x/y rather than a group transform, so contrast checkers can see the
        // panel behind the label. Only the dot fades; the label stays readable.
        dot.setAttribute('cx', String(sx(p[0])))
        dot.setAttribute('cy', String(sy(p[1])))
        label.setAttribute('x', String(sx(p[0])))
        label.setAttribute('y', String(sy(p[1]) - 14))
        const chosen = run.phase === 'route' && run.decision === 'ESCALATE' && run.strategy === strategy
        dot.style.opacity = chosen ? '1' : '0.45'
      })
      frame = requestAnimationFrame(tick)
    }
    tick()
    return () => cancelAnimationFrame(frame)
  }, [sample, animateMotion])

  return (
    <svg viewBox="0 0 640 320" className="h-full w-full" role="img" aria-label="Gate routing diagram">
      <line x1={sx(LAYOUT.start[0])} y1={sy(LAYOUT.start[1])} x2={sx(gate[0])} y2={sy(gate[1])} stroke="#F4B942" strokeOpacity="0.45" strokeDasharray="5 5" />
      <path d={`M${pt(gate)} Q${sx(gate[0] + 1.2)},${sy((gate[1] + cotPos[1]) / 2)} ${pt(cotPos)}`} stroke="#B7D9C4" strokeOpacity="0.8" fill="none" strokeWidth="2" />
      <path d={`M${pt(gate)} Q${sx((gate[0] + core[0]) / 2)},${sy(gate[1] + 1.1)} ${pt(core)}`} stroke="#E0643A" strokeOpacity="0.8" fill="none" strokeWidth="2" />

      <g>
        <circle cx={sx(gate[0])} cy={sy(gate[1])} r="46" fill="none" stroke="#F4B942" strokeWidth="1.5" />
        <circle cx={sx(gate[0])} cy={sy(gate[1])} r="34" fill="#F4B942" fillOpacity="0.08" stroke="#F4B942" strokeOpacity="0.5" />
        <circle ref={cot} cx={sx(cotPos[0])} cy={sy(cotPos[1])} r="16" fill="#B7D9C4" style={{ transition: 'opacity .3s' }} />
        <polygon
          points={[0, 1, 2, 3, 4, 5].map((i) => `${sx(core[0]) + 22 * Math.cos((i * Math.PI) / 3)},${sy(core[1]) + 22 * Math.sin((i * Math.PI) / 3)}`).join(' ')}
          fill="#E0643A"
          fillOpacity="0.9"
        />
        {STRATEGIES.map((strategy, i) => (
          <g key={strategy}>
            <circle ref={(el) => {
              agents.current[i] = el
            }} r="9" fill="#F6E27A" style={{ transition: 'opacity .3s' }} />
            <text ref={(el) => {
              agentLabels.current[i] = el
            }} textAnchor="middle" fill="#FFFCF7" className="font-mono text-[11px]">
              {STRATEGY_LABELS[strategy]}
            </text>
          </g>
        ))}
        <circle ref={orb} r="8" fill="#F5F0E8" style={{ opacity: 0 }} />
      </g>
      <text x={sx(gate[0])} y={sy(gate[1]) - 58} textAnchor="middle" fill="#F4B942" className="font-mono text-[12px]">Gate</text>
      <text x={sx(cotPos[0])} y={sy(cotPos[1]) + 32} textAnchor="middle" fill="#B7D9C4" className="font-mono text-[12px]">CoT-SC answer</text>
      <text x={sx(core[0])} y={sy(core[1] - 1.62)} textAnchor="middle" fill="#F4A07E" className="font-mono text-[12px]">Orchestrator</text>
    </svg>
  )
}

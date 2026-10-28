import type { Strategy } from '../lib/types'

// Shared layout and timing for the 3D gate scene and its 2D fallback, so both
// tell the same story.

export type Vec3 = [number, number, number]
export type Decision = 'STOP' | 'ESCALATE'

export const LAYOUT = {
  start: [-5, 0.7, 0] as Vec3,
  gate: [-0.6, 0.7, 0] as Vec3,
  cot: [-0.6, -2.3, 0] as Vec3,
  core: [3.4, 0.7, 0] as Vec3,
  agentRadius: 1.65,
}

export const STRATEGIES: Strategy[] = ['react', 'debate', 'reflexion']

export const TIMING = { approach: 1.4, hold: 0.7, route: 1.5, hop: 0.8 }

/** Angle of each agent on its orbit around the core at time t (seconds). */
export function agentPosition(index: number, t: number): Vec3 {
  const angle = t * 0.35 + (index * Math.PI * 2) / STRATEGIES.length
  const [cx, cy, cz] = LAYOUT.core
  return [
    cx + Math.cos(angle) * LAYOUT.agentRadius,
    cy + Math.sin(angle) * LAYOUT.agentRadius * 0.75,
    cz + Math.sin(angle) * 0.6,
  ]
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t
const ease = (t: number) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2)
const mix = (a: Vec3, b: Vec3, t: number): Vec3 => [
  lerp(a[0], b[0], t),
  lerp(a[1], b[1], t),
  lerp(a[2], b[2], t),
]

/** Quadratic Bezier between two points with a control point. */
function bezier(a: Vec3, c: Vec3, b: Vec3, t: number): Vec3 {
  return mix(mix(a, c, t), mix(c, b, t), t)
}

export type Phase = 'idle' | 'approach' | 'waiting' | 'route'

/**
 * Where the task orb is, given the run state.
 *
 * - waiting: the request is in flight; the orb circles the gate.
 * - route:   a decision arrived `elapsed` seconds ago; the orb follows its path.
 */
export function orbPosition(
  phase: Phase,
  elapsed: number,
  decision: Decision | null,
  strategy: Strategy | null,
  clock: number,
): { position: Vec3; visible: boolean; atTarget: boolean } {
  const { start, gate, cot, core } = LAYOUT
  if (phase === 'idle') return { position: start, visible: false, atTarget: false }
  if (phase === 'approach') {
    return { position: mix(start, gate, ease(Math.min(elapsed / TIMING.approach, 1))), visible: true, atTarget: false }
  }
  if (phase === 'waiting') {
    const t = Math.min(elapsed / TIMING.approach, 1)
    if (t < 1) return { position: mix(start, gate, ease(t)), visible: true, atTarget: false }
    const a = clock * 2.4
    return {
      position: [gate[0] + Math.cos(a) * 0.35, gate[1] + Math.sin(a) * 0.35, gate[2]],
      visible: true,
      atTarget: false,
    }
  }
  // route
  if (decision === 'STOP') {
    const t = Math.min(elapsed / TIMING.route, 1)
    const control: Vec3 = [gate[0] + 1.2, (gate[1] + cot[1]) / 2, 0.4]
    return { position: bezier(gate, control, cot, ease(t)), visible: true, atTarget: t >= 1 }
  }
  const toCore = Math.min(elapsed / TIMING.route, 1)
  if (toCore < 1) {
    const control: Vec3 = [(gate[0] + core[0]) / 2, gate[1] + 1.1, 0.3]
    return { position: bezier(gate, control, core, ease(toCore)), visible: true, atTarget: false }
  }
  const index = strategy ? STRATEGIES.indexOf(strategy) : -1
  if (index < 0) return { position: core, visible: true, atTarget: true }
  const hop = Math.min((elapsed - TIMING.route) / TIMING.hop, 1)
  return { position: mix(core, agentPosition(index, clock), ease(hop)), visible: true, atTarget: hop >= 1 }
}

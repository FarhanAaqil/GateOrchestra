import { useEffect, useRef } from 'react'
import type { Strategy } from '../lib/types'
import { STRATEGIES, TIMING, type Decision, type Phase } from './gateTimeline'

export interface GateRunInput {
  /** A request is in flight. */
  running: boolean
  /** Decision of the latest completed run, or null. */
  decision: Decision | null
  /** Sub-agent that handled an escalation, if any. */
  strategy: Strategy | null
  /** Changes whenever a new run starts (restarts the animation). */
  runKey: number
  /** Loop an illustrative STOP/ESCALATE cycle when there is no live run. */
  demo: boolean
}

export interface GateRunState {
  phase: Phase
  decision: Decision | null
  strategy: Strategy | null
  elapsed: number
}

const DEMO_PERIOD = TIMING.approach + TIMING.hold + TIMING.route + TIMING.hop + 2.6

/**
 * Turns run props into an animation state sampled every frame. Returns a
 * function so render loops (useFrame / rAF) can read it without re-rendering.
 */
export function useGateRun(input: GateRunInput): (now: number) => GateRunState {
  const startedAt = useRef<number | null>(null)
  const decidedAt = useRef<number | null>(null)
  const inputRef = useRef(input)

  useEffect(() => {
    inputRef.current = input
  })

  useEffect(() => {
    startedAt.current = input.running || input.decision ? performance.now() / 1000 : null
    decidedAt.current = null
  }, [input.runKey]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (input.decision && !input.running) {
      const now = performance.now() / 1000
      // Let the orb reach the gate before it is routed.
      const reached = (startedAt.current ?? now) + TIMING.approach + TIMING.hold
      decidedAt.current = Math.max(now, reached)
    }
  }, [input.decision, input.running, input.runKey])

  return (now: number): GateRunState => {
    const { running, decision, strategy, demo } = inputRef.current
    if (startedAt.current !== null && (running || decision)) {
      const elapsed = now - startedAt.current
      if (running || decidedAt.current === null || now < decidedAt.current) {
        return { phase: 'waiting', decision: null, strategy: null, elapsed }
      }
      return { phase: 'route', decision, strategy, elapsed: now - decidedAt.current }
    }
    if (!demo) return { phase: 'idle', decision: null, strategy: null, elapsed: 0 }

    // Illustrative loop: alternate STOP and ESCALATE through each sub-agent.
    const cycle = Math.floor(now / DEMO_PERIOD)
    const t = now - cycle * DEMO_PERIOD
    const demoDecision: Decision = cycle % 2 === 0 ? 'STOP' : 'ESCALATE'
    const demoStrategy = STRATEGIES[Math.floor(cycle / 2) % STRATEGIES.length]
    if (t < TIMING.approach) return { phase: 'approach', decision: null, strategy: null, elapsed: t }
    if (t < TIMING.approach + TIMING.hold) return { phase: 'waiting', decision: null, strategy: null, elapsed: TIMING.approach }
    return {
      phase: 'route',
      decision: demoDecision,
      strategy: demoDecision === 'ESCALATE' ? demoStrategy : null,
      elapsed: t - TIMING.approach - TIMING.hold,
    }
  }
}

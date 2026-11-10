import type { MethodId } from './types'

export interface MethodInfo {
  id: MethodId
  label: string
  /** One sentence for a reader who has never seen the paper. */
  summary: string
}

export const METHODS: MethodInfo[] = [
  {
    id: 'GateOrchestra',
    label: 'GateOrchestra (recommended)',
    summary: 'A quick check first. A trained gate then decides whether the answer is good enough or the agent team should take over.',
  },
  {
    id: 'RuleBasedGate',
    label: 'Rule-based gate',
    summary: 'Same quick check, but a few fixed rules decide when to call the agent team. Simple, and a useful point of comparison.',
  },
  {
    id: 'RandomGate',
    label: 'Random gate (baseline)',
    summary: 'Flips a coin to decide whether to call the agent team. Only here to show what chance looks like.',
  },
  {
    id: 'CoT-SC',
    label: 'Quick answer only',
    summary: 'Asks the model three times and takes the majority answer (chain-of-thought self-consistency). Cheap, never uses the agent team.',
  },
  {
    id: 'Always-MAS',
    label: 'Always use the agent team',
    summary: 'Skips the quick check and always runs the multi-agent system (ReAct, Debate or Reflexion). The most expensive option.',
  },
]

export function methodInfo(id: MethodId): MethodInfo {
  return METHODS.find((m) => m.id === id) ?? METHODS[0]
}

/** What the agent team may spend, in words, for the budget multiplier k. */
export function budgetLabel(k: number): string {
  return `${k}× the quick check${k === 3 ? ' (standard)' : k < 3 ? ' (tighter)' : ' (more room)'}`
}

/** Why Run is disabled right now, or null when it can run. Shown next to the button. */
export function runBlockReason(question: string, backend: string, running: boolean): string | null {
  if (running) return null
  if (backend === 'locked') return 'Sign in with the API token first.'
  if (backend !== 'online') return 'Waiting for the API to come online.'
  if (!question.trim()) return 'Type a question, or pick a benchmark question, to run it.'
  return null
}

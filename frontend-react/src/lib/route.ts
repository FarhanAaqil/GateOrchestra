import { GATE_LABELS, type TurnMeta } from './chat'

export interface RouteSummary {
  /** Plain-language outcome shown on the answer. */
  headline: string
  escalated: boolean
  /** One sentence for the tooltip. */
  explanation: string
  /** Label/value rows for the "how this was answered" panel. */
  details: { label: string; value: string }[]
}

/** "groq:openai/gpt-oss-20b" -> "gpt-oss-20b (groq)"; plain names pass through. */
export function formatModel(raw: string): string {
  const match = raw.match(/^(groq|gemini|ollama|mock):(.+)$/)
  const provider = match ? match[1] : null
  const name = (match ? match[2] : raw).split('/').pop() ?? raw
  return provider ? `${name} (${provider})` : name
}

export function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`
}

/** Turn the gate's technical outcome into wording a non-expert can read. */
export function describeRoute(meta: Partial<TurnMeta>): RouteSummary | null {
  if (!meta.gate) return null
  const escalated = meta.decision === 'ESCALATE'
  const gate = GATE_LABELS[meta.gate] ?? meta.gate
  const headline = meta.cached ? 'Instant answer (cached)' : escalated ? 'Researched with tools' : 'Answered directly'
  const explanation = escalated
    ? 'The quick check was not confident enough, so a research agent with tools worked on it.'
    : 'The quick check was confident, so the question was answered directly without extra work.'
  const details: { label: string; value: string }[] = [{ label: 'Routing', value: `${gate} chose ${escalated ? 'to research' : 'to answer directly'}` }]
  if (meta.gate_reason) details.push({ label: 'Why', value: meta.gate_reason })
  if (meta.task_type) details.push({ label: 'Question type', value: meta.task_type })
  if (meta.consistency != null) details.push({ label: 'Quick-check agreement', value: `${Math.round(meta.consistency * 100)}%` })
  if (meta.tokens) {
    details.push({
      label: 'Tokens',
      value: `${meta.tokens.probe.toLocaleString()} check · ${meta.tokens.agent.toLocaleString()} research · ${meta.tokens.answer.toLocaleString()} answer`,
    })
  }
  if (meta.models) {
    for (const [role, model] of Object.entries(meta.models)) details.push({ label: `Model (${role})`, value: formatModel(model) })
  }
  return { headline, escalated, explanation, details }
}

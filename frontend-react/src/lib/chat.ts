import { API_BASE_URL, clearApiCache } from './api'

export type GateName = 'learned' | 'rule_based' | 'always_stop' | 'always_escalate' | 'triage'

export interface ToolCall {
  name: string
  input: Record<string, unknown>
  output: string
  round: number
}

export interface TurnMeta {
  gate: GateName
  decision: 'STOP' | 'ESCALATE'
  gate_reason: string
  task_type: string
  consistency: number
  tokens: { probe: number; agent: number; answer: number }
  total_tokens: number
  models: Record<string, string>
  tools: ToolCall[]
  latency_ms: number
  cached?: boolean
  stopped?: boolean
  verdicts?: Record<string, string>
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  meta?: Partial<TurnMeta>
  rating?: number | null
  created_at?: number
}

export interface Conversation {
  id: string
  title: string
  updated_at: number
}

export interface GateScore {
  gate: GateName
  escalate_rate: number
  expected_tokens: number
  up: number
  down: number
  accuracy: number
  eligible: boolean
  chosen: boolean
}

export interface KeyAttempt {
  provider: string
  reason: 'rate_limited' | 'invalid_key' | 'no_key' | 'unavailable' | string
}

export type ChatEvent =
  | { type: 'meta'; conversation_id: string; user_message_id: string }
  | { type: 'probe'; task_type: string; consistency: number; samples: string[]; answer: string; tokens: number; latency_ms: number }
  | { type: 'gate'; gate: GateName; decision: 'STOP' | 'ESCALATE'; reason: string; verdicts: Record<string, string> }
  | ({ type: 'tool' } & ToolCall)
  | { type: 'answer_start'; model_role: string }
  | { type: 'delta'; text: string }
  | { type: 'done'; message_id: string; conversation_id: string; meta: TurnMeta }
  | { type: 'error'; code: string; message: string; needs_key?: boolean; attempts?: KeyAttempt[] }

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init)
  const data = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = (data as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status})`)
  }
  return data as T
}

/** Send one message and receive the turn's events as they stream in. */
export async function streamChat(
  body: { message: string; conversation_id?: string | null; gate_mode?: string | null },
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => null)
    const detail = (data as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Chat failed (${response.status})`)
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let split = buffer.indexOf('\n\n')
    while (split >= 0) {
      const frame = buffer.slice(0, split)
      buffer = buffer.slice(split + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as ChatEvent)
      }
      split = buffer.indexOf('\n\n')
    }
  }
  clearApiCache()
}

export const chatApi = {
  conversations: () => json<{ conversations: Conversation[] }>('/conversations'),
  conversation: (id: string) => json<Conversation & { messages: ChatMessage[] }>(`/conversations/${id}`),
  rename: (id: string, title: string) =>
    json(`/conversations/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) }),
  remove: (id: string) => json(`/conversations/${id}`, { method: 'DELETE' }),
  rate: (messageId: string, rating: -1 | 0 | 1) =>
    json<{ message_id: string; rating: number }>(`/messages/${messageId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rating }),
    }),
  gates: () => json<{ margin: number; explore: number; gates: GateScore[] }>('/assistant/gates'),
  addKey: (provider: 'groq' | 'gemini', key: string) =>
    json<{ provider: string; key: string; models: number }>('/assistant/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, key }),
    }),
}

export const GATE_LABELS: Record<GateName, string> = {
  learned: 'Learned gate',
  rule_based: 'Rule-based gate',
  always_stop: 'Always stop',
  always_escalate: 'Always escalate',
  triage: 'Triage',
}

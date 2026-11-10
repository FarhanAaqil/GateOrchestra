import { describe, expect, it } from 'vitest'
import type { Conversation, TurnMeta } from './chat'
import { filterConversations, groupConversations } from './conversations'
import { describeRoute, formatModel, formatSeconds } from './route'

const NOW = new Date(2026, 10, 10, 15, 0, 0).getTime() // 10 Nov 2026, 15:00 local

function convo(id: string, title: string, hoursAgo: number): Conversation {
  return { id, title, updated_at: (NOW - hoursAgo * 3_600_000) / 1000 }
}

describe('groupConversations', () => {
  it('groups by recency, newest first, and drops empty groups', () => {
    const groups = groupConversations(
      [
        convo('old', 'Old one', 24 * 30),
        convo('today-a', 'Earlier today', 5),
        convo('yesterday', 'Yesterday one', 20),
        convo('today-b', 'Just now', 0.1),
        convo('week', 'This week', 24 * 4),
      ],
      NOW,
    )
    expect(groups.map((g) => g.label)).toEqual(['Today', 'Yesterday', 'Previous 7 days', 'Older'])
    expect(groups[0].items.map((c) => c.id)).toEqual(['today-b', 'today-a'])
    expect(groups[1].items.map((c) => c.id)).toEqual(['yesterday'])
  })

  it('returns no groups for no conversations', () => {
    expect(groupConversations([], NOW)).toEqual([])
  })

  it('omits labels that have no conversations', () => {
    const groups = groupConversations([convo('a', 'Only today', 1)], NOW)
    expect(groups.map((g) => g.label)).toEqual(['Today'])
  })
})

describe('filterConversations', () => {
  const items = [convo('1', 'Nile versus Amazon length', 1), convo('2', 'Python hash map example', 2)]

  it('matches every word, ignoring case', () => {
    expect(filterConversations(items, 'amazon NILE').map((c) => c.id)).toEqual(['1'])
  })

  it('returns everything for a blank query', () => {
    expect(filterConversations(items, '   ')).toHaveLength(2)
  })

  it('returns nothing when no title matches', () => {
    expect(filterConversations(items, 'tokyo')).toEqual([])
  })
})

describe('describeRoute', () => {
  const base: Partial<TurnMeta> = {
    gate: 'learned',
    decision: 'STOP',
    gate_reason: 'consistent probe',
    task_type: 'factual',
    consistency: 1,
    tokens: { probe: 600, agent: 0, answer: 669 },
    models: { answer: 'openai/gpt-oss-120b' },
  }

  it('says "Answered directly" when the gate stopped', () => {
    const route = describeRoute(base)!
    expect(route.headline).toBe('Answered directly')
    expect(route.escalated).toBe(false)
    expect(route.details.find((d) => d.label === 'Quick-check agreement')?.value).toBe('100%')
  })

  it('says "Researched with tools" when the gate escalated', () => {
    const route = describeRoute({ ...base, decision: 'ESCALATE' })!
    expect(route.headline).toBe('Researched with tools')
    expect(route.escalated).toBe(true)
  })

  it('prefers the cached wording', () => {
    expect(describeRoute({ ...base, cached: true })!.headline).toBe('Instant answer (cached)')
  })

  it('has nothing to say without a gate', () => {
    expect(describeRoute({})).toBeNull()
  })

  it('shows models without the provider prefix noise', () => {
    expect(formatModel('groq:openai/gpt-oss-20b')).toBe('gpt-oss-20b (groq)')
    expect(formatModel('qwen2.5:3b')).toBe('qwen2.5:3b')
    expect(formatModel('ollama:qwen2.5:3b')).toBe('qwen2.5:3b (ollama)')
    expect(formatModel('gemini-2.5-flash')).toBe('gemini-2.5-flash')
    const route = describeRoute({ ...base, models: { answer: 'groq:openai/gpt-oss-20b' } })!
    expect(route.details.find((d) => d.label === 'Model (answer)')?.value).toBe('gpt-oss-20b (groq)')
  })

  it('formats durations in seconds', () => {
    expect(formatSeconds(22_380)).toBe('22.4 s')
  })
})

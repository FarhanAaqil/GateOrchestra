import type { Conversation } from './chat'

export interface ConversationGroup {
  label: string
  items: Conversation[]
}

const DAY_MS = 86_400_000

function startOfDay(ms: number): number {
  const d = new Date(ms)
  d.setHours(0, 0, 0, 0)
  return d.getTime()
}

/** Keep conversations whose title contains every word of the query (case-insensitive). */
export function filterConversations(items: Conversation[], query: string): Conversation[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return items
  return items.filter((c) => words.every((w) => c.title.toLowerCase().includes(w)))
}

/**
 * Group conversations by when they were last used (Today, Yesterday, Previous 7 days, Older),
 * newest first. `updated_at` is in seconds, as the API returns it.
 */
export function groupConversations(items: Conversation[], now: number = Date.now()): ConversationGroup[] {
  const today = startOfDay(now)
  const buckets: ConversationGroup[] = [
    { label: 'Today', items: [] },
    { label: 'Yesterday', items: [] },
    { label: 'Previous 7 days', items: [] },
    { label: 'Older', items: [] },
  ]
  const sorted = [...items].sort((a, b) => b.updated_at - a.updated_at)
  for (const c of sorted) {
    const when = c.updated_at * 1000
    if (when >= today) buckets[0].items.push(c)
    else if (when >= today - DAY_MS) buckets[1].items.push(c)
    else if (when >= today - 7 * DAY_MS) buckets[2].items.push(c)
    else buckets[3].items.push(c)
  }
  return buckets.filter((b) => b.items.length)
}

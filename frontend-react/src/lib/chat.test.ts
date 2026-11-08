import { expect, it, vi } from 'vitest'
import { streamChat, type ChatEvent } from './chat'

function sseResponse(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream({
    start(controller) {
      chunks.forEach((c) => controller.enqueue(encoder.encode(c)))
      controller.close()
    },
  })
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}

it('parses events even when frames are split across network chunks', async () => {
  const frames = [
    { type: 'meta', conversation_id: 'c1', user_message_id: 'u1' },
    { type: 'delta', text: 'Hello ' },
    { type: 'delta', text: 'world' },
  ].map((e) => `data: ${JSON.stringify(e)}\n\n`).join('')
  const cut = Math.floor(frames.length / 2)
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(sseResponse([frames.slice(0, cut), frames.slice(cut)]))
  const events: ChatEvent[] = []
  await streamChat({ message: 'hi' }, (e) => events.push(e))
  expect(events.map((e) => e.type)).toEqual(['meta', 'delta', 'delta'])
  expect(events.filter((e) => e.type === 'delta').map((e) => (e as { text: string }).text).join('')).toBe('Hello world')
})

it('surfaces the API error message when the request is refused', async () => {
  vi.spyOn(globalThis, 'fetch').mockResolvedValue(
    new Response(JSON.stringify({ detail: 'The assistant is busy' }), { status: 503 }),
  )
  await expect(streamChat({ message: 'hi' }, () => undefined)).rejects.toThrow('The assistant is busy')
})

import { beforeEach, describe, expect, it, vi } from 'vitest'
import { api, checkBackend, clearApiCache } from './api'

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

describe('api client', () => {
  beforeEach(() => clearApiCache())

  it('shares one request between concurrent identical GETs', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => jsonResponse({ status: 'verified' }))
    const [a, b] = await Promise.all([api.summary(), api.summary()])
    expect(a).toEqual(b)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await api.summary()
    expect(fetchMock).toHaveBeenCalledTimes(1) // still cached
  })

  it('does not cache failures, so the next call retries', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(jsonResponse({ detail: 'boom' }, 500))
      .mockResolvedValueOnce(jsonResponse({ status: 'verified' }))
    await expect(api.summary()).rejects.toThrow('boom')
    await expect(api.summary()).resolves.toEqual({ status: 'verified' })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears the cache after a run, so pages see the new history', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => jsonResponse({ ok: true }))
    await api.models()
    await api.run({ task_id: 't', question: 'q', method: 'CoT-SC', k: 3, force_simulation: true })
    await api.models()
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it('never caches history', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => jsonResponse({ count: 0, history: [] }))
    await api.history()
    await api.history()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('rejects a server on the port that is not GateOrchestra', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(jsonResponse({ name: 'Some other app' }))
    await expect(checkBackend()).rejects.toThrow(/not the GateOrchestra API/)
  })
})

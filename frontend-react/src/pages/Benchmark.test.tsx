import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import { clearApiCache } from '../lib/api'
import Benchmark from './Benchmark'

function task(id: string, question: string) {
  return { task_id: id, question, depth_score: 2, parallel_score: 1, source_dataset: 'hotpotqa' }
}

// 120 train tasks force a second page; val and test have one task each.
const TRAIN = Array.from({ length: 120 }, (_, i) => task(`train_${i}`, `Train question ${i}`))
const SPLITS: Record<string, ReturnType<typeof task>[]> = {
  train: TRAIN,
  val: [task('val_0', 'Which river is longest?')],
  test: [task('test_0', 'Who painted the ceiling?')],
}

beforeEach(() => {
  clearApiCache()
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input) => {
    const url = new URL(String(input))
    const split = url.searchParams.get('split')!
    const limit = Number(url.searchParams.get('limit'))
    const offset = Number(url.searchParams.get('offset'))
    const rows = SPLITS[split]
    const body = { split, total: rows.length, limit, offset, tasks: rows.slice(offset, offset + limit) }
    return new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })
  })
})

it('loads every page of every split', async () => {
  render(<Benchmark />)
  expect(await screen.findByText('122 of 122 tasks')).toBeInTheDocument()
  expect(screen.getByText('train_119')).toBeInTheDocument()
})

it('filters by search only after typing pauses', async () => {
  render(<Benchmark />)
  await screen.findByText('122 of 122 tasks')
  vi.useFakeTimers()
  try {
    fireEvent.change(screen.getByPlaceholderText('id or question'), { target: { value: 'river' } })
    expect(screen.getByText('122 of 122 tasks')).toBeInTheDocument()
    await act(async () => vi.advanceTimersByTime(200))
    expect(screen.getByText('1 of 122 tasks')).toBeInTheDocument()
    expect(screen.getByText('val_0')).toBeInTheDocument()
  } finally {
    vi.useRealTimers()
  }
})

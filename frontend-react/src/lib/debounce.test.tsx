import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { useDebouncedValue } from './debounce'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

it('updates only after the value stops changing', () => {
  const { result, rerender } = renderHook(({ value }) => useDebouncedValue(value, 200), {
    initialProps: { value: 'a' },
  })
  rerender({ value: 'ab' })
  act(() => vi.advanceTimersByTime(150))
  rerender({ value: 'abc' })
  act(() => vi.advanceTimersByTime(150))
  expect(result.current).toBe('a')
  act(() => vi.advanceTimersByTime(60))
  expect(result.current).toBe('abc')
})

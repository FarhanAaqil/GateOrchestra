import { expect, it } from 'vitest'
import { isCitable, num, pct } from './format'

it('formats missing values as a dash', () => {
  expect(pct(null)).toBe('—')
  expect(num(undefined)).toBe('—')
  expect(pct(12.345)).toBe('12.3%')
})

it('treats only verified or partial results as citable', () => {
  expect(isCitable('verified')).toBe(true)
  expect(isCitable('partial')).toBe(true)
  expect(isCitable('unverified')).toBe(false)
  expect(isCitable(undefined)).toBe(false)
})

import { describe, expect, it } from 'vitest'
import { accuracyDomain } from './domain'

describe('accuracyDomain', () => {
  it('zooms in when every method is accurate', () => {
    expect(accuracyDomain([92.2, 96.9, 97.9])).toEqual([85, 100])
  })

  it('never goes below zero', () => {
    expect(accuracyDomain([52, 60])).toEqual([45, 100])
    expect(accuracyDomain([3, 99])).toEqual([0, 100])
  })

  it('keeps the full range when some method is poor', () => {
    expect(accuracyDomain([30, 90])).toEqual([0, 100])
  })

  it('has a sensible default with no data', () => {
    expect(accuracyDomain([])).toEqual([0, 100])
  })
})

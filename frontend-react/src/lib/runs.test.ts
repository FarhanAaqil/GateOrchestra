import { describe, expect, it } from 'vitest'
import { GLOSSARY } from './glossary'
import { friendlyRunLabel } from './runs'

describe('friendlyRunLabel', () => {
  it('turns a timestamped id into a readable label', () => {
    expect(friendlyRunLabel('ui-1794080322070')).toMatch(/^Run on .+ at .+/)
  })

  it('names demo runs', () => {
    expect(friendlyRunLabel('live_demo_2')).toBe('Demo run 2')
  })

  it('leaves benchmark task ids and unknown ids alone', () => {
    expect(friendlyRunLabel('arith_004')).toBe('arith_004')
    expect(friendlyRunLabel('ui-abc')).toBe('ui-abc')
  })
})

describe('GLOSSARY', () => {
  it('explains every term in a full sentence', () => {
    for (const text of Object.values(GLOSSARY)) expect(text.length).toBeGreaterThan(40)
  })
})

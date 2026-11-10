import { describe, expect, it } from 'vitest'
import { METHODS, budgetLabel, methodInfo, runBlockReason } from './methods'

describe('methods', () => {
  it('describes every method in plain words', () => {
    expect(METHODS.map((m) => m.id)).toEqual(['GateOrchestra', 'RuleBasedGate', 'RandomGate', 'CoT-SC', 'Always-MAS'])
    for (const m of METHODS) {
      expect(m.label.length).toBeLessThan(32) // short enough not to be cut off in the form
      expect(m.summary.length).toBeGreaterThan(30)
    }
  })

  it('looks a method up by id', () => {
    expect(methodInfo('CoT-SC').label).toBe('Quick answer only')
  })
})

describe('budgetLabel', () => {
  it('names the standard budget and the tighter and looser ones', () => {
    expect(budgetLabel(3)).toBe('3× the quick check (standard)')
    expect(budgetLabel(2)).toBe('2× the quick check (tighter)')
    expect(budgetLabel(5)).toBe('5× the quick check (more room)')
  })
})

describe('runBlockReason', () => {
  it('asks for a question first', () => {
    expect(runBlockReason('   ', 'online', false)).toMatch(/Type a question/)
  })

  it('explains an offline or locked backend before the question', () => {
    expect(runBlockReason('Who?', 'error', false)).toMatch(/API/)
    expect(runBlockReason('Who?', 'locked', false)).toMatch(/Sign in/)
  })

  it('has no reason when it can run, or while it is already running', () => {
    expect(runBlockReason('Who painted it?', 'online', false)).toBeNull()
    expect(runBlockReason('', 'online', true)).toBeNull()
  })
})

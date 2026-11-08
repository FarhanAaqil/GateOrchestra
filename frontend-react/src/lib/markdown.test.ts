import { expect, it } from 'vitest'
import { normalizeMath } from './markdown'

it('turns \\( \\) and \\[ \\] into dollar math but leaves money and code alone', () => {
  const input = 'Cost is $5,000. Inline \\(x^2\\) and display:\n\\[ A = P(1+r)^n \\]\n`\\(code\\)`'
  const out = normalizeMath(input)
  expect(out).toContain('$5,000')
  expect(out).toContain('$$x^2$$')
  expect(out).toContain('\n$$\nA = P(1+r)^n\n$$\n')
  expect(out).toContain('`\\(code\\)`')
})

it('drops citation placeholders such as 【source】', () => {
  expect(normalizeMath('Capacity is 88,966【source】.')).toBe('Capacity is 88,966.')
})

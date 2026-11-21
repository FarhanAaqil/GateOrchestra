// Accessibility check with axe-core: serious and critical WCAG 2.1 A/AA problems fail the test.
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const AXE = require.resolve('axe-core/axe.min.js')

/** Run axe on the current page; returns readable descriptions of serious/critical violations. */
export async function axeProblems(page) {
  await page.addScriptTag({ path: AXE })
  const result = await page.evaluate(async () => {
    // Animations settle first so colours are measured at rest.
    await new Promise((r) => setTimeout(r, 400))
    return await window.axe.run(document, {
      runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
      resultTypes: ['violations'],
    })
  })
  return result.violations
    .filter((v) => v.impact === 'serious' || v.impact === 'critical')
    .map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(' ')).join(', ')}`)
}

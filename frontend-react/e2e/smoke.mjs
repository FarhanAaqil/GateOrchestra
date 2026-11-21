// End-to-end smoke test: a real browser drives the built console against the
// API running the offline mock provider (no model calls, no keys).
//   CHROME_PATH=/usr/bin/google-chrome BASE_URL=http://localhost:4173 node e2e/smoke.mjs
import puppeteer from 'puppeteer-core'
import { axeProblems } from './a11y.mjs'

const BASE = process.env.BASE_URL ?? 'http://localhost:4173'
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
  defaultViewport: { width: 1280, height: 860 },
})
const errors = []
let failed = false

async function check(name, fn) {
  try {
    await fn()
    console.log(`PASS ${name}`)
  } catch (err) {
    failed = true
    console.log(`FAIL ${name}: ${err.message}`)
  }
}

try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => errors.push(e.message))

  await check('console reaches the API', async () => {
    await page.goto(`${BASE}/chat`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => document.body.innerText.includes('Backend online'), { timeout: 20000 })
  })

  await check('a chat turn streams an answer with routing badges', async () => {
    await page.type('textarea[aria-label="Message"]', 'What is the answer to everything?')
    await page.keyboard.press('Enter')
    await page.waitForFunction(() => document.body.innerText.includes('Offline answer'), { timeout: 20000 })
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Stop"]'), { timeout: 20000 })
    const text = await page.evaluate(() => document.body.innerText)
    if (!/Answered directly|Researched with tools|Instant answer/.test(text)) throw new Error('no routing badge')
    if (!text.includes('How this was answered')) throw new Error('no explanation panel')
    if (!(await page.$('.prose-chat pre code'))) throw new Error('code block not rendered')
  })

  await check('the chat shows which model answers, not the benchmark model', async () => {
    await page.waitForFunction(() => document.body.innerText.includes('Chat models'), { timeout: 10000 })
  })

  await check('the conversation appears in the history and can be renamed', async () => {
    await page.waitForSelector('button[aria-label^="Rename "]', { timeout: 10000 })
    await page.click('button[aria-label^="Rename "]', { delay: 50 })
    await page.waitForSelector('input[aria-label="Conversation title"]', { timeout: 5000 })
    await page.click('input[aria-label="Conversation title"]', { clickCount: 3 })
    await page.type('input[aria-label="Conversation title"]', 'Renamed by the smoke test')
    await page.keyboard.press('Enter')
    await page.waitForFunction(() => document.body.innerText.includes('Renamed by the smoke test'), { timeout: 10000 })
  })

  await check('regenerate replaces the last answer instead of adding one', async () => {
    await page.waitForSelector('button[aria-label="Regenerate answer"]', { timeout: 5000 })
    await page.click('button[aria-label="Regenerate answer"]')
    await page.waitForFunction(() => document.body.innerText.includes('Offline answer'), { timeout: 20000 })
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Stop"]'), { timeout: 20000 })
    const answers = await page.$$eval('button[aria-label="Copy answer"]', (els) => els.length)
    if (answers !== 1) throw new Error(`expected 1 answer after regenerate, found ${answers}`)
  })

  await check('an answer can be rated', async () => {
    await page.click('button[aria-label="Helpful"]')
    await page.waitForSelector('button[aria-label="Helpful"][aria-pressed="true"]', { timeout: 5000 })
  })

  await check('an earlier question can be edited and sent again', async () => {
    await page.hover('button[aria-label="Edit this question"]').catch(() => undefined)
    await page.click('button[aria-label="Edit this question"]')
    await page.waitForSelector('textarea[aria-label="Edit your question"]', { timeout: 5000 })
    await page.click('textarea[aria-label="Edit your question"]', { clickCount: 3 })
    await page.keyboard.down('Control')
    await page.keyboard.press('KeyA')
    await page.keyboard.up('Control')
    await page.type('textarea[aria-label="Edit your question"]', 'An edited question')
    await page.keyboard.press('Enter')
    // Wait for the new answer itself, not just the edited question (which shows at once).
    await page.waitForFunction(
      () =>
        document.body.innerText.includes('An edited question') &&
        !document.querySelector('button[aria-label="Stop"]') &&
        document.querySelectorAll('button[aria-label="Copy answer"]').length === 1,
      { timeout: 20000 },
    )
    if ((await page.evaluate(() => document.body.innerText)).includes('What is the answer to everything?')) throw new Error('the old question is still shown')
  })

  await check('the admin page counts the turn', async () => {
    await page.goto(`${BASE}/admin`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => /Chat turns\s*[1-9]/.test(document.body.innerText), { timeout: 15000 })
  })

  await check('operator pages are accessible', async () => {
    for (const path of ['/chat', '/admin', '/run', '/tokens']) {
      await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle0' })
      const problems = await axeProblems(page)
      if (problems.length) throw new Error(`${path}: ${problems.join(' | ')}`)
    }
  })

  if (errors.length) {
    failed = true
    console.log(`FAIL page errors: ${errors.join(' | ')}`)
  }
} finally {
  await browser.close()
}
process.exit(failed ? 1 : 0)

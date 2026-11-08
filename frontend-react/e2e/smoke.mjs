// End-to-end smoke test: a real browser drives the built console against the
// API running the offline mock provider (no model calls, no keys).
//   CHROME_PATH=/usr/bin/google-chrome BASE_URL=http://localhost:4173 node e2e/smoke.mjs
import puppeteer from 'puppeteer-core'

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
    if (!/STOP|ESCALATE/.test(text)) throw new Error('no routing badge')
    if (!(await page.$('.prose-chat pre code'))) throw new Error('code block not rendered')
  })

  await check('an answer can be rated', async () => {
    await page.click('button[aria-label="Helpful"]')
    await page.waitForSelector('button[aria-label="Helpful"][aria-pressed="true"]', { timeout: 5000 })
  })

  await check('the admin page counts the turn', async () => {
    await page.goto(`${BASE}/admin`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => /Chat turns\s*[1-9]/.test(document.body.innerText), { timeout: 15000 })
  })

  if (errors.length) {
    failed = true
    console.log(`FAIL page errors: ${errors.join(' | ')}`)
  }
} finally {
  await browser.close()
}
process.exit(failed ? 1 : 0)

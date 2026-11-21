// End-to-end test of hosted (multi-user) mode: a real browser against the API started by
// scripts/e2e_hosted_server.py (offline mock model, emails written to a file).
//   CHROME_PATH=... BASE_URL=http://127.0.0.1:4174 MAIL_LOG=/tmp/e2e-hosted/mail.log node e2e/hosted.mjs
import { createHmac } from 'node:crypto'
import { readFileSync } from 'node:fs'
import puppeteer from 'puppeteer-core'
import { axeProblems } from './a11y.mjs'

const BASE = process.env.BASE_URL ?? 'http://127.0.0.1:4174'
const MAIL_LOG = process.env.MAIL_LOG ?? '/tmp/e2e-hosted/mail.log'
const PASSWORD = 'e2e password long enough'
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

/** The newest link of a kind (verify / reset) emailed to an address. */
function linkFor(email, kind = 'verify') {
  const blocks = readFileSync(MAIL_LOG, 'utf8').split('----').filter((b) => b.includes(`TO ${email}`))
  const match = blocks.reverse().map((b) => b.match(new RegExp(`(https?://\\S+/${kind}\\?token=[\\w-]+)`))).find(Boolean)
  if (!match) throw new Error(`no ${kind} link for ${email}`)
  return match[1].replace(/^https?:\/\/[^/]+/, BASE)
}

/** RFC 6238 code for a base32 secret at a 30-second step (what an authenticator app shows). */
function totp(secret, step = Math.floor(Date.now() / 30000)) {
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
  let bits = ''
  for (const ch of secret.toUpperCase()) bits += alphabet.indexOf(ch).toString(2).padStart(5, '0')
  const key = Buffer.from(bits.match(/.{8}/g).map((b) => parseInt(b, 2)))
  const counter = Buffer.alloc(8)
  counter.writeBigUInt64BE(BigInt(step))
  const digest = createHmac('sha1', key).update(counter).digest()
  const offset = digest[digest.length - 1] & 15
  return String((digest.readUInt32BE(offset) & 0x7fffffff) % 1_000_000).padStart(6, '0')
}

const text = (page) => page.evaluate(() => document.body.innerText)
const waitText = (page, s, timeout = 15000) => page.waitForFunction((t) => document.body.innerText.includes(t), { timeout }, s)

async function signUpAndConfirm(page, email) {
  await page.goto(`${BASE}/signup`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('#email', { timeout: 10000 })
  await page.type('#email', email)
  await page.type('#password', PASSWORD)
  await page.click('input[type="checkbox"]')
  await page.click('button[type="submit"]')
  await waitText(page, 'Check')
  await page.goto(linkFor(email), { waitUntil: 'networkidle0' })
  await waitText(page, 'Your address is confirmed')
}

async function logIn(page, email, next = '/chat') {
  await page.goto(`${BASE}/login?next=${encodeURIComponent(next)}`, { waitUntil: 'networkidle0' })
  await page.waitForSelector('#email', { timeout: 10000 })
  await page.type('#email', email)
  await page.type('#password', PASSWORD)
  await page.click('button[type="submit"]')
  await page.waitForFunction((n) => location.pathname === n, { timeout: 15000 }, next)
}

async function logOut(page) {
  await page.waitForSelector('button[aria-label="Log out"]', { timeout: 10000 })
  await page.evaluate(() => document.querySelector('button[aria-label="Log out"]').click())
  try {
    await page.waitForFunction(() => location.pathname === '/' && !document.querySelector('button[aria-label="Log out"]'), { timeout: 15000 })
  } catch {
    throw new Error(`log out did not reach the home page signed out (on ${await page.evaluate(() => location.pathname)})`)
  }
}

async function accessible(page, label) {
  const problems = await axeProblems(page)
  if (problems.length) throw new Error(`${label}: ${problems.join(' | ')}`)
}

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
  defaultViewport: { width: 1280, height: 860 },
})
const errors = []

try {
  const page = await browser.newPage()
  page.on('pageerror', (e) => errors.push(e.message))

  await check('a visitor sees the product page and no operator pages', async () => {
    await page.goto(BASE, { waitUntil: 'networkidle0' })
    await waitText(page, 'knows how hard to think')
    const nav = await page.$eval('nav[aria-label="Main"]', (n) => n.innerText)
    for (const hidden of ['Admin', 'Run a Task', 'Token Accountant']) if (nav.includes(hidden)) throw new Error(`${hidden} is visible`)
    await accessible(page, 'landing')
  })

  await check('typing /admin while signed out leads home', async () => {
    await page.goto(`${BASE}/admin`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => location.pathname === '/', { timeout: 10000 })
  })

  await check('the chat asks for a login', async () => {
    await page.goto(`${BASE}/chat`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => location.pathname === '/login', { timeout: 10000 })
    await accessible(page, 'login')
  })

  await check('a person signs up, confirms, logs in and chats', async () => {
    await signUpAndConfirm(page, 'ada@example.com')
    await logIn(page, 'ada@example.com')
    await page.waitForSelector('textarea[aria-label="Message"]', { timeout: 15000 })
    await page.type('textarea[aria-label="Message"]', 'What is the answer to everything?')
    await page.keyboard.press('Enter')
    await waitText(page, 'Offline answer', 20000)
    await page.waitForFunction(() => !document.querySelector('button[aria-label="Stop"]'), { timeout: 20000 })
    if (!/Answered directly|Researched with tools|Instant answer/.test(await text(page))) throw new Error('no routing label')
    await accessible(page, 'chat')
  })

  await check('usage counts the answer against the plan', async () => {
    await page.goto(`${BASE}/usage`, { waitUntil: 'networkidle0' })
    await page.waitForFunction(() => /of 30 messages today/.test(document.body.innerText), { timeout: 10000 })
    if (!/Messages\s*1\b/.test(await text(page))) throw new Error('the message was not counted')
    await accessible(page, 'usage')
  })

  await check('a normal user cannot open operator pages', async () => {
    const nav = await page.$eval('nav[aria-label="Main"]', (n) => n.innerText)
    if (nav.includes('Admin')) throw new Error('Admin is in the menu')
    for (const path of ['/admin', '/run', '/tokens']) {
      await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle0' })
      await page.waitForFunction(() => location.pathname === '/', { timeout: 10000 })
    }
  })

  await check('two-factor sign-in can be switched on and is then required', async () => {
    await page.goto(`${BASE}/settings`, { waitUntil: 'networkidle0' })
    await page.waitForSelector('input[aria-label="Password to set up two-factor sign-in"]', { timeout: 10000 })
    await page.type('input[aria-label="Password to set up two-factor sign-in"]', PASSWORD)
    await page.click('xpath/.//button[normalize-space()="Set up"]')
    await page.waitForSelector('img[alt="QR code for your authenticator app"]', { timeout: 10000 })
    const secret = await page.$eval('p.break-all', (el) => el.textContent.trim())
    const step = Math.floor(Date.now() / 30000)
    await page.type('input[aria-label="Code from your authenticator app"]', totp(secret, step))
    await page.click('xpath/.//button[normalize-space()="Turn on"]')
    await waitText(page, 'Save these recovery codes now')
    await logOut(page)
    await page.goto(`${BASE}/login`, { waitUntil: 'networkidle0' })
    await page.waitForSelector('#email', { timeout: 10000 })
    await page.type('#email', 'ada@example.com')
    await page.type('#password', PASSWORD)
    await page.click('button[type="submit"]')
    await waitText(page, 'Two-factor sign-in')
    await page.type('#code', totp(secret, step + 1)) // the next code: the first one cannot be replayed
    await page.click('button[type="submit"]')
    await page.waitForFunction(() => location.pathname === '/chat', { timeout: 15000 })
  })

  await check('settings, pricing and legal pages are accessible', async () => {
    for (const path of ['/settings', '/pricing', '/privacy']) {
      await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle0' })
      await accessible(page, path)
    }
    await logOut(page)
  })

  await check('an admin can switch a plan on after a payment', async () => {
    await signUpAndConfirm(page, 'admin@example.com')
    await logIn(page, 'admin@example.com', '/admin')
    await waitText(page, 'People and plans')
    const row = await page.waitForFunction(() => [...document.querySelectorAll('tr')].find((r) => r.innerText.includes('ada@example.com')), { timeout: 10000 })
    const button = await row.$('button')
    await button.click()
    await page.waitForFunction(
      () => [...document.querySelectorAll('tr')].some((r) => r.innerText.includes('ada@example.com') && /\bpro\b/.test(r.innerText)),
      { timeout: 10000 },
    )
    await accessible(page, 'admin')
  })

  await check('research pages stay public and accessible', async () => {
    await logOut(page)
    for (const path of ['/overview', '/results', '/benchmark', '/baselines', '/gate', '/about']) {
      await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle0' })
      await accessible(page, path)
    }
  })

  await check('the layout works on a phone', async () => {
    await page.setViewport({ width: 375, height: 760 })
    for (const path of ['/', '/login', '/pricing']) {
      await page.goto(`${BASE}${path}`, { waitUntil: 'networkidle0' })
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
      if (overflow > 1) throw new Error(`${path} scrolls sideways by ${overflow}px at 375px`)
    }
    await page.click('button[aria-label="Open menu"]')
    await page.waitForSelector('button[aria-label="Close menu"]', { timeout: 5000 })
  })

  if (errors.length) {
    failed = true
    console.log(`FAIL page errors: ${errors.join(' | ')}`)
  }
} finally {
  await browser.close()
}
process.exit(failed ? 1 : 0)

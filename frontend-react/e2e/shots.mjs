// Screenshots of the main pages for design review (not part of CI).
//   node e2e/shots.mjs <base-url> <out-dir> [cookie-header]
import puppeteer from 'puppeteer-core'
import { mkdirSync } from 'node:fs'

const [base = 'http://localhost:3700', out = 'shots', cookie = ''] = process.argv.slice(2)
const pages = (process.env.SHOT_PAGES || '/,/login,/signup,/pricing,/about').split(',')
mkdirSync(out, { recursive: true })
const browser = await puppeteer.launch({ executablePath: process.env.CHROME_PATH ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true, args: ['--no-sandbox'] })
try {
  for (const [w, h, tag] of [[1440, 900, 'desk'], [390, 844, 'phone']]) {
    const page = await browser.newPage()
    await page.setViewport({ width: w, height: h })
    if (cookie) {
      const host = new URL(base).hostname
      await page.setCookie(...cookie.split(';').map((c) => {
        const [name, ...v] = c.trim().split('=')
        return { name, value: v.join('='), domain: host, path: '/' }
      }))
    }
    for (const path of pages) {
      await page.goto(base + path, { waitUntil: 'networkidle2', timeout: 30000 }).catch(() => {})
      await new Promise((r) => setTimeout(r, 900))
      const name = (path === '/' ? 'home' : path.replace(/\//g, '_').replace(/^_/, '')) + '-' + tag
      await page.screenshot({ path: `${out}/${name}.png`, fullPage: tag === 'desk' })
    }
    await page.close()
  }
} finally {
  await browser.close()
}
console.log('done')


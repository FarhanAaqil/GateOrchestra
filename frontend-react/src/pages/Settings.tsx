import { Download, KeyRound, Loader2, Lock, Mail, ShieldCheck, Trash2, UserRound } from 'lucide-react'
import QRCode from 'qrcode'
import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Card, ErrorNote, Page, PageHeader } from '../components/ui'
import { MIN_PASSWORD, accountApi, profileApi, securityApi, usageApi, type KeyProvider } from '../lib/account'
import { useApp } from '../lib/context'
import { FIELD } from '../lib/styles'
import { useApi } from '../lib/useApi'

const PROVIDER_LABEL: Record<KeyProvider, string> = { groq: 'Groq', gemini: 'Google Gemini', openai: 'OpenAI-compatible' }
const KEY_HINT: Record<KeyProvider, string> = { groq: 'gsk_…', gemini: 'AIza…', openai: 'sk-…' }

function OwnKeys() {
  const keys = useApi(accountApi.keys)
  const [provider, setProvider] = useState<KeyProvider>('groq')
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const save = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await accountApi.putKey(provider, key.trim())
      setKey('')
      keys.reload()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const remove = async (name: string) => {
    await accountApi.deleteKey(name).catch((err: Error) => setError(err.message))
    keys.reload()
  }

  return (
    <Card tilt={false}>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><KeyRound className="h-4 w-4 text-muted" /> Your own API key</h2>
      <p className="mt-1 text-sm text-muted">
        Optional. With your own Groq, Gemini or OpenAI-compatible key your chats run on your account with that provider and do not use up your
        plan's daily tokens. The key is checked with the provider, stored encrypted, and never shown again, not even to you.
      </p>
      {keys.error ? <div className="mt-3"><ErrorNote message={keys.error} /></div> : null}
      <ul className="mt-3 space-y-1.5 text-sm">
        {keys.data?.keys.length ? (
          keys.data.keys.map((k) => (
            <li key={k.provider} className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-moss" />
              <span className="font-semibold text-ink">{PROVIDER_LABEL[k.provider]}</span>
              <span className="text-muted">added {new Date(k.added_at * 1000).toLocaleDateString()}</span>
              <button onClick={() => remove(k.provider)} className="ml-auto inline-flex items-center gap-1 rounded-md border border-wine/40 px-2.5 py-0.5 text-xs text-wine hover:bg-wine/10" aria-label={`Remove your ${PROVIDER_LABEL[k.provider]} key`}>
                <Trash2 className="h-3 w-3" /> Remove
              </button>
            </li>
          ))
        ) : (
          <li className="text-muted">{keys.loading ? 'Loading…' : 'No key of your own: the service’s keys answer your chats.'}</li>
        )}
      </ul>
      <form onSubmit={save} className="mt-4 grid gap-2 sm:grid-cols-[auto_minmax(0,1fr)_auto]">
        <select value={provider} onChange={(e) => setProvider(e.target.value as KeyProvider)} className="rounded-lg border border-line bg-paper px-3 py-2 text-sm" aria-label="Provider">
          <option value="groq">Groq</option>
          <option value="gemini">Google Gemini</option>
          <option value="openai">OpenAI-compatible</option>
        </select>
        <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(e) => setKey(e.target.value)} placeholder={KEY_HINT[provider]} className={`${FIELD} font-mono text-sm`} aria-label="API key" />
        <button type="submit" disabled={!key.trim() || busy} className="inline-flex items-center justify-center gap-2 rounded-lg bg-forest px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-forest-soft disabled:opacity-40">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Save key
        </button>
      </form>
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

/** Two-factor sign-in: scan a QR code, confirm a code, keep the recovery codes. */
function TwoFactor() {
  const { user, setUser } = useApp()
  const status = useApi(securityApi.status)
  const [password, setPassword] = useState('')
  const [setup, setSetup] = useState<{ secret: string; uri: string } | null>(null)
  const [qr, setQr] = useState<string | null>(null)
  const [code, setCode] = useState('')
  const [recovery, setRecovery] = useState<string[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!setup) return
    QRCode.toDataURL(setup.uri, { margin: 1, width: 192, color: { dark: '#0B4A6B', light: '#FFFFFF' } }).then(setQr, () => setQr(null))
  }, [setup])

  const run = async (action: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try {
      await action()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  const start = (e: FormEvent) => {
    e.preventDefault()
    run(async () => {
      const r = await securityApi.setup(password)
      setSetup({ secret: r.secret, uri: r.otpauth_uri })
      setPassword('')
    })
  }
  const confirm = (e: FormEvent) => {
    e.preventDefault()
    run(async () => {
      const r = await securityApi.enable(code.trim())
      setRecovery(r.recovery_codes)
      setSetup(null)
      setCode('')
      if (user) setUser({ ...user, two_factor: true })
      status.reload()
    })
  }
  const turnOff = (e: FormEvent) => {
    e.preventDefault()
    run(async () => {
      await securityApi.disable(password, code.trim())
      setPassword('')
      setCode('')
      setRecovery(null)
      if (user) setUser({ ...user, two_factor: false })
      status.reload()
    })
  }

  const on = status.data?.two_factor
  return (
    <Card>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><ShieldCheck className="h-4 w-4 text-muted" /> Two-factor sign-in</h2>
      <p className="mt-1 text-sm text-muted">
        {on
          ? `On. ${status.data?.recovery_codes_left ?? 0} recovery code${status.data?.recovery_codes_left === 1 ? '' : 's'} left.`
          : 'Off. With it on, signing in also needs a code from an authenticator app such as 1Password, Google Authenticator or Authy.'}
      </p>
      {recovery ? (
        <div className="mt-3 rounded-lg border border-amber/40 bg-amber/10 p-3" role="status">
          <p className="text-sm font-medium text-ink">Save these recovery codes now. Each works once if you lose your phone, and they are not shown again.</p>
          <ul className="mt-2 grid grid-cols-2 gap-1 font-mono text-sm text-ink">{recovery.map((c) => <li key={c}>{c}</li>)}</ul>
        </div>
      ) : null}
      {!on && !setup ? (
        <form onSubmit={start} className="mt-3 flex flex-wrap gap-2">
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Your password" aria-label="Password to set up two-factor sign-in" className={`${FIELD} max-w-xs text-sm`} />
          <button type="submit" disabled={!password || busy} className="btn btn-primary">Set up</button>
        </form>
      ) : null}
      {setup ? (
        <form onSubmit={confirm} className="mt-3 grid gap-3 sm:grid-cols-[auto_minmax(0,1fr)]">
          {qr ? <img src={qr} alt="QR code for your authenticator app" className="h-40 w-40 rounded-lg border border-line" /> : null}
          <div className="space-y-2 text-sm text-muted">
            <p>Scan the code with your authenticator app, or enter this key by hand:</p>
            <p className="break-all rounded-md bg-forest/5 px-2 py-1 font-mono text-xs text-ink">{setup.secret}</p>
            <div className="flex flex-wrap gap-2">
              <input inputMode="numeric" autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} placeholder="6-digit code" aria-label="Code from your authenticator app" className={`${FIELD} max-w-[10rem] text-sm`} />
              <button type="submit" disabled={code.trim().length < 6 || busy} className="btn btn-primary">Turn on</button>
            </div>
          </div>
        </form>
      ) : null}
      {on ? (
        <form onSubmit={turnOff} className="mt-3 flex flex-wrap gap-2">
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" aria-label="Password to turn off two-factor sign-in" className={`${FIELD} max-w-[12rem] text-sm`} />
          <input autoComplete="one-time-code" value={code} onChange={(e) => setCode(e.target.value)} placeholder="Code" aria-label="Authenticator or recovery code" className={`${FIELD} max-w-[9rem] text-sm`} />
          <button type="submit" disabled={!password || code.trim().length < 6 || busy} className="btn btn-secondary">Turn off</button>
        </form>
      ) : null}
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

function ChangeEmail() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sent, setSent] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await securityApi.changeEmail(email.trim(), password)
      setSent(true)
      setPassword('')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><Mail className="h-4 w-4 text-muted" /> Email address</h2>
      {sent ? (
        <p className="mt-2 text-sm text-muted" role="status">Check {email.trim()} for a link to confirm it. Your address changes only when you open that link.</p>
      ) : (
        <form onSubmit={submit} className="mt-3 grid gap-2">
          <input type="email" autoComplete="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="New email address" aria-label="New email address" className={`${FIELD} text-sm`} />
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Your password" aria-label="Password to change email" className={`${FIELD} text-sm`} />
          <button type="submit" disabled={!email.trim() || !password || busy} className="btn btn-primary justify-self-start">Send confirmation link</button>
        </form>
      )}
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

function Profile() {
  const { user } = useApp()
  const usage = useApi(usageApi.usage)
  const [error, setError] = useState<string | null>(null)
  const download = async () => {
    setError(null)
    try {
      const blob = await profileApi.exportData()
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'gateorchestra-export.json'
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError((err as Error).message)
    }
  }
  if (!user) return null
  const plan = usage.data?.plan
  return (
    <Card tilt={false}>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><UserRound className="h-4 w-4 text-muted" /> Profile and plan</h2>
      <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted">Email</dt>
        <dd className="break-all text-ink">{user.email}{user.verified ? '' : ' (not confirmed yet)'}</dd>
        <dt className="text-muted">Member since</dt>
        <dd className="text-ink">{new Date(user.created_at * 1000).toLocaleDateString()}</dd>
        <dt className="text-muted">Plan</dt>
        <dd className="text-ink">
          {plan ? plan.label : '…'}
          {usage.data?.paid_until ? ` until ${new Date(usage.data.paid_until * 1000).toLocaleDateString()}` : ''}
          {' · '}<Link to="/pricing" className="underline">plans</Link>
          {' · '}<Link to="/usage" className="underline">usage</Link>
        </dd>
      </dl>
      <button onClick={download} className="mt-4 inline-flex items-center gap-1.5 rounded-lg border border-line px-3.5 py-1.5 text-sm text-forest hover:bg-forest/5">
        <Download className="h-4 w-4" /> Export my conversations
      </button>
      <p className="mt-1 text-xs text-muted">A JSON file with your conversations, the names of your documents and your account details.</p>
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

function ChangePassword() {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setDone(false)
    if (next.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters.`)
      return
    }
    setBusy(true)
    setError(null)
    try {
      await profileApi.changePassword(current, next)
      setCurrent('')
      setNext('')
      setDone(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card tilt={false}>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><Lock className="h-4 w-4 text-muted" /> Password</h2>
      <form onSubmit={submit} className="mt-3 grid gap-2">
        <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} placeholder="Current password" aria-label="Current password" className={`${FIELD} text-sm`} />
        <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} placeholder={`New password (${MIN_PASSWORD}+ characters)`} aria-label="New password" className={`${FIELD} text-sm`} />
        <button type="submit" disabled={!current || !next || busy} className="justify-self-start rounded-lg bg-forest px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-forest-soft disabled:opacity-40">Change password</button>
      </form>
      <p className="mt-1 text-xs text-muted">Your other browsers and devices are signed out; this one stays signed in.</p>
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
      {done ? <p className="mt-2 text-sm text-forest" role="status">Password changed.</p> : null}
    </Card>
  )
}

function DeleteAccount() {
  const { setUser } = useApp()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await profileApi.deleteAccount(password)
      setUser(null)
      navigate('/', { replace: true })
    } catch (err) {
      setError((err as Error).message)
      setBusy(false)
    }
  }
  return (
    <Card tilt={false}>
      <h2 className="flex items-center gap-2 text-lg font-semibold text-wine"><Trash2 className="h-5 w-5" /> Delete my account</h2>
      <p className="mt-1 text-sm text-muted">
        Removes your account and everything in it: conversations, messages, uploaded documents, cached answers, ratings, your own
        API keys and your usage history. It cannot be undone, so export first if you want a copy.
      </p>
      {open ? (
        <form onSubmit={submit} className="mt-3 flex flex-wrap gap-2">
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Your password" aria-label="Password to confirm deletion" className={`${FIELD} max-w-xs text-sm`} />
          <button type="submit" disabled={!password || busy} className="inline-flex items-center gap-2 rounded-lg bg-wine px-4 py-2 text-sm font-semibold text-canvas disabled:opacity-40">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Delete everything
          </button>
          <button type="button" onClick={() => setOpen(false)} className="rounded-lg border border-line px-4 py-2 text-sm text-forest">Cancel</button>
        </form>
      ) : (
        <button onClick={() => setOpen(true)} className="mt-3 rounded-lg border border-wine/50 px-4 py-2 text-sm text-wine hover:bg-wine/10">Delete my account…</button>
      )}
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

export default function Settings() {
  const { user } = useApp()
  return (
    <Page>
      <PageHeader kicker="Account" title="Your settings">
        Signed in as {user?.email}.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-2">
        <Profile />
        <ChangePassword />
        <TwoFactor />
        <ChangeEmail />
        <div className="lg:col-span-2"><OwnKeys /></div>
        <div className="lg:col-span-2"><DeleteAccount /></div>
      </div>
    </Page>
  )
}

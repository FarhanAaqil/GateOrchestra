import { Download, KeyRound, Loader2, Lock, Trash2, UserRound } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Card, ErrorNote, Page, PageHeader } from '../components/ui'
import { MIN_PASSWORD, accountApi, profileApi, usageApi } from '../lib/account'
import { useApp } from '../lib/context'
import { FIELD } from '../lib/styles'
import { useApi } from '../lib/useApi'

const PROVIDER_LABEL = { groq: 'Groq', gemini: 'Google Gemini' } as const

function OwnKeys() {
  const keys = useApi(accountApi.keys)
  const [provider, setProvider] = useState<'groq' | 'gemini'>('groq')
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
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><KeyRound className="h-4 w-4 text-navy/50" /> Your own API key</h2>
      <p className="mt-1 text-sm text-navy/70">
        Optional. With your own Groq or Gemini key your chats run on your account with that provider and do not use up your
        plan's daily tokens. The key is checked with the provider, stored encrypted, and never shown again, not even to you.
      </p>
      {keys.error ? <div className="mt-3"><ErrorNote message={keys.error} /></div> : null}
      <ul className="mt-3 space-y-1.5 text-sm">
        {keys.data?.keys.length ? (
          keys.data.keys.map((k) => (
            <li key={k.provider} className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-cyan" />
              <span className="font-semibold text-ink">{PROVIDER_LABEL[k.provider]}</span>
              <span className="text-navy/60">added {new Date(k.added_at * 1000).toLocaleDateString()}</span>
              <button onClick={() => remove(k.provider)} className="ml-auto inline-flex items-center gap-1 rounded-md border border-wine/40 px-2.5 py-0.5 text-xs text-wine hover:bg-wine/10" aria-label={`Remove your ${PROVIDER_LABEL[k.provider]} key`}>
                <Trash2 className="h-3 w-3" /> Remove
              </button>
            </li>
          ))
        ) : (
          <li className="text-navy/60">{keys.loading ? 'Loading…' : 'No key of your own: the service’s keys answer your chats.'}</li>
        )}
      </ul>
      <form onSubmit={save} className="mt-4 grid gap-2 sm:grid-cols-[auto_minmax(0,1fr)_auto]">
        <select value={provider} onChange={(e) => setProvider(e.target.value as 'groq' | 'gemini')} className="rounded-lg border border-navy/15 bg-ice px-3 py-2 text-sm" aria-label="Provider">
          <option value="groq">Groq</option>
          <option value="gemini">Google Gemini</option>
        </select>
        <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(e) => setKey(e.target.value)} placeholder={provider === 'groq' ? 'gsk_…' : 'AIza…'} className={`${FIELD} font-mono text-sm`} aria-label="API key" />
        <button type="submit" disabled={!key.trim() || busy} className="inline-flex items-center justify-center gap-2 rounded-lg bg-ink px-4 py-2 text-sm font-semibold text-soft disabled:opacity-40">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Save key
        </button>
      </form>
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
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><UserRound className="h-4 w-4 text-navy/50" /> Profile and plan</h2>
      <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1 text-sm">
        <dt className="text-navy/60">Email</dt>
        <dd className="break-all text-ink">{user.email}{user.verified ? '' : ' (not confirmed yet)'}</dd>
        <dt className="text-navy/60">Member since</dt>
        <dd className="text-ink">{new Date(user.created_at * 1000).toLocaleDateString()}</dd>
        <dt className="text-navy/60">Plan</dt>
        <dd className="text-ink">
          {plan ? plan.label : '…'}
          {usage.data?.paid_until ? ` until ${new Date(usage.data.paid_until * 1000).toLocaleDateString()}` : ''}
          {' · '}<Link to="/pricing" className="underline">plans</Link>
          {' · '}<Link to="/usage" className="underline">usage</Link>
        </dd>
      </dl>
      <button onClick={download} className="mt-4 inline-flex items-center gap-1.5 rounded-lg border border-navy/15 px-3.5 py-1.5 text-sm text-navy hover:bg-navy/5">
        <Download className="h-4 w-4" /> Export my conversations
      </button>
      <p className="mt-1 text-xs text-navy/60">A JSON file with your conversations, the names of your documents and your account details.</p>
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
      <h2 className="flex items-center gap-2 text-lg font-semibold text-ink"><Lock className="h-4 w-4 text-navy/50" /> Password</h2>
      <form onSubmit={submit} className="mt-3 grid gap-2">
        <input type="password" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} placeholder="Current password" aria-label="Current password" className={`${FIELD} text-sm`} />
        <input type="password" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} placeholder={`New password (${MIN_PASSWORD}+ characters)`} aria-label="New password" className={`${FIELD} text-sm`} />
        <button type="submit" disabled={!current || !next || busy} className="justify-self-start rounded-lg bg-ink px-4 py-2 text-sm font-semibold text-soft disabled:opacity-40">Change password</button>
      </form>
      <p className="mt-1 text-xs text-navy/60">Your other browsers and devices are signed out; this one stays signed in.</p>
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
      {done ? <p className="mt-2 text-sm text-navy" role="status">Password changed.</p> : null}
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
      <p className="mt-1 text-sm text-navy/70">
        Removes your account and everything in it: conversations, messages, uploaded documents, cached answers, ratings, your own
        API keys and your usage history. It cannot be undone, so export first if you want a copy.
      </p>
      {open ? (
        <form onSubmit={submit} className="mt-3 flex flex-wrap gap-2">
          <input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Your password" aria-label="Password to confirm deletion" className={`${FIELD} max-w-xs text-sm`} />
          <button type="submit" disabled={!password || busy} className="inline-flex items-center gap-2 rounded-lg bg-wine px-4 py-2 text-sm font-semibold text-soft disabled:opacity-40">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Delete everything
          </button>
          <button type="button" onClick={() => setOpen(false)} className="rounded-lg border border-navy/15 px-4 py-2 text-sm text-navy">Cancel</button>
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
      <PageHeader number="10" kicker="Account" title={<>Your settings</>}>
        Signed in as {user?.email}.
      </PageHeader>
      <div className="grid gap-4 lg:grid-cols-2">
        <Profile />
        <ChangePassword />
        <div className="lg:col-span-2"><OwnKeys /></div>
        <div className="lg:col-span-2"><DeleteAccount /></div>
      </div>
    </Page>
  )
}

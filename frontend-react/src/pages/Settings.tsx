import { KeyRound, Loader2, Trash2 } from 'lucide-react'
import { useState, type FormEvent } from 'react'
import { Card, ErrorNote, Page, PageHeader } from '../components/ui'
import { accountApi } from '../lib/account'
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
      <h2 className="flex items-center gap-2 text-lg font-bold text-ink"><KeyRound className="h-5 w-5 text-wine" /> Your own API key</h2>
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
              <button onClick={() => remove(k.provider)} className="ml-auto inline-flex items-center gap-1 rounded-full border border-wine/40 px-2.5 py-0.5 text-xs text-wine hover:bg-wine/10" aria-label={`Remove your ${PROVIDER_LABEL[k.provider]} key`}>
                <Trash2 className="h-3 w-3" /> Remove
              </button>
            </li>
          ))
        ) : (
          <li className="text-navy/60">{keys.loading ? 'Loading…' : 'No key of your own: the service’s keys answer your chats.'}</li>
        )}
      </ul>
      <form onSubmit={save} className="mt-4 grid gap-2 sm:grid-cols-[auto_minmax(0,1fr)_auto]">
        <select value={provider} onChange={(e) => setProvider(e.target.value as 'groq' | 'gemini')} className="rounded-2xl border border-sky bg-ice px-3 py-2 text-sm" aria-label="Provider">
          <option value="groq">Groq</option>
          <option value="gemini">Google Gemini</option>
        </select>
        <input type="password" autoComplete="off" spellCheck={false} value={key} onChange={(e) => setKey(e.target.value)} placeholder={provider === 'groq' ? 'gsk_…' : 'AIza…'} className={`${FIELD} font-mono text-sm`} aria-label="API key" />
        <button type="submit" disabled={!key.trim() || busy} className="inline-flex items-center justify-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-bold text-soft disabled:opacity-40">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null} Save key
        </button>
      </form>
      {error ? <p className="mt-2 text-sm text-wine" role="alert">{error}</p> : null}
    </Card>
  )
}

export default function Settings() {
  const { user } = useApp()
  return (
    <Page>
      <PageHeader number="10" kicker="Account" title={<>Your <span className="mark-lemon">settings</span></>}>
        Signed in as {user?.email}.
      </PageHeader>
      <div className="grid gap-4">
        <OwnKeys />
      </div>
    </Page>
  )
}

import { KeyRound, Loader2, X } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { chatApi, type KeyAttempt } from '../lib/chat'
import { FIELD } from '../lib/styles'

const REASON: Record<string, string> = {
  rate_limited: 'hit its rate limit',
  invalid_key: 'rejected the key',
  no_key: 'has no key yet',
  unavailable: 'is over capacity or unreachable',
}

/** Asks for another Groq or Gemini key when every configured provider has failed. */
export default function KeyDialog({
  attempts,
  onClose,
  onSaved,
}: {
  attempts: KeyAttempt[]
  onClose: () => void
  onSaved: (provider: string) => void
}) {
  const groqFailed = attempts.some((a) => a.provider === 'groq' && a.reason !== 'no_key')
  const [provider, setProvider] = useState<'groq' | 'gemini'>(groqFailed ? 'gemini' : 'groq')
  const [key, setKey] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    input.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (!key.trim()) return
    setSaving(true)
    setError(null)
    try {
      await chatApi.addKey(provider, key.trim())
      setKey('')
      onSaved(provider)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/50 p-4 backdrop-blur-sm" role="presentation" onClick={onClose}>
      <form
        role="dialog"
        aria-modal="true"
        aria-labelledby="key-dialog-title"
        onSubmit={submit}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md rounded-3xl bg-soft p-6 shadow-2xl"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="rounded-full bg-lemon p-2 text-ink"><KeyRound className="h-5 w-5" /></span>
            <h2 id="key-dialog-title" className="text-xl font-bold text-ink">Another API key is needed</h2>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-1 text-navy/60 hover:bg-ice" aria-label="Close">
            <X className="h-5 w-5" />
          </button>
        </div>

        <ul className="mb-4 space-y-1 text-sm text-navy/80">
          {attempts.length ? (
            attempts.map((a, i) => (
              <li key={i}>
                <span className="font-semibold capitalize">{a.provider}</span> {REASON[a.reason] ?? a.reason}.
              </li>
            ))
          ) : (
            <li>No model provider is configured.</li>
          )}
        </ul>

        <fieldset className="mb-3 grid grid-cols-2 gap-2">
          <legend className="mb-1.5 text-sm font-medium text-navy">Provider</legend>
          {(['groq', 'gemini'] as const).map((p) => (
            <label
              key={p}
              className={`cursor-pointer rounded-2xl border px-3 py-2 text-sm ${provider === p ? 'border-navy bg-navy text-ice' : 'border-sky text-navy hover:bg-ice'}`}
            >
              <input type="radio" name="provider" value={p} checked={provider === p} onChange={() => setProvider(p)} className="sr-only" />
              <span className="font-semibold">{p === 'groq' ? 'Groq' : 'Google Gemini'}</span>
              <span className="block text-xs opacity-75">{p === 'groq' ? 'starts with gsk_' : 'from Google AI Studio'}</span>
            </label>
          ))}
        </fieldset>

        <label className="text-sm font-medium text-navy" htmlFor="api-key">API key</label>
        <input
          id="api-key"
          ref={input}
          type="password"
          autoComplete="off"
          spellCheck={false}
          value={key}
          onChange={(e) => setKey(e.target.value)}
          placeholder={provider === 'groq' ? 'gsk_…' : 'AIza…'}
          className={`${FIELD} mt-1.5 font-mono text-sm`}
        />
        <p className="mt-2 text-xs text-navy/60">
          The key is checked with {provider === 'groq' ? 'Groq' : 'Google'}, then saved only to this server's <code>.env</code>. It is never shown again.
        </p>
        {error ? <p className="mt-3 rounded-xl bg-wine/10 px-3 py-2 text-sm text-wine" role="alert">{error}</p> : null}

        <div className="mt-5 flex justify-end gap-2">
          <button type="button" onClick={onClose} className="rounded-full border border-sky px-4 py-2 text-sm text-navy hover:bg-ice">
            Cancel
          </button>
          <button
            type="submit"
            disabled={!key.trim() || saving}
            className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-2 text-sm font-bold text-soft disabled:opacity-40"
          >
            {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
            {saving ? 'Checking…' : 'Save and retry'}
          </button>
        </div>
      </form>
    </div>
  )
}

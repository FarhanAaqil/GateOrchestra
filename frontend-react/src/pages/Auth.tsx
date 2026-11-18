import { motion } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { Link, Navigate, useNavigate, useSearchParams } from 'react-router-dom'
import { Page } from '../components/ui'
import { MIN_PASSWORD, authApi } from '../lib/account'
import { useApp } from '../lib/context'
import { rise } from '../lib/motion'
import { FIELD } from '../lib/styles'

/** Only same-site paths may be used as a post-login destination. */
function safeNext(raw: string | null): string {
  return raw && raw.startsWith('/') && !raw.startsWith('//') ? raw : '/chat'
}

function AuthCard({ title, intro, children }: { title: string; intro?: ReactNode; children: ReactNode }) {
  return (
    <Page>
      <motion.div variants={rise} className="card mx-auto mt-6 w-full max-w-md p-6 md:p-8">
        <h1 className="text-3xl font-bold text-ink">{title}</h1>
        {intro ? <div className="mt-2 text-sm text-navy/70">{intro}</div> : null}
        <div className="mt-6">{children}</div>
      </motion.div>
    </Page>
  )
}

function Field({ id, label, type = 'text', value, onChange, autoComplete, hint }: {
  id: string
  label: string
  type?: string
  value: string
  onChange: (value: string) => void
  autoComplete?: string
  hint?: string
}) {
  return (
    <div className="mb-4">
      <label htmlFor={id} className="text-sm font-medium text-navy">{label}</label>
      <input
        id={id}
        type={type}
        value={value}
        autoComplete={autoComplete}
        onChange={(e) => onChange(e.target.value)}
        className={`${FIELD} mt-1.5 text-sm`}
        aria-describedby={hint ? `${id}-hint` : undefined}
        required
      />
      {hint ? <p id={`${id}-hint`} className="mt-1 text-xs text-navy/60">{hint}</p> : null}
    </div>
  )
}

function Submit({ busy, children, disabled }: { busy: boolean; children: ReactNode; disabled?: boolean }) {
  return (
    <button
      type="submit"
      disabled={busy || disabled}
      className="inline-flex w-full items-center justify-center gap-2 rounded-full bg-ink px-5 py-2.5 text-sm font-bold text-soft disabled:opacity-40"
    >
      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
      {children}
    </button>
  )
}

function Problem({ message }: { message: string | null }) {
  return message ? <p className="mb-4 rounded-xl bg-wine/10 px-3 py-2 text-sm text-wine" role="alert">{message}</p> : null
}

function Done({ children }: { children: ReactNode }) {
  return <p className="rounded-xl bg-cyan/20 px-3 py-3 text-sm text-navy" role="status">{children}</p>
}

/** Pages for single-tenant servers, which have no accounts. */
function NoAccounts() {
  return (
    <AuthCard title="No accounts here" intro="This server runs in single-tenant mode, so there is nothing to sign in to.">
      <Link to="/chat" className="text-sm font-semibold text-navy underline">Go to the chat</Link>
    </AuthCard>
  )
}

export function Login() {
  const { hosted, user, setUser } = useApp()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  if (!hosted) return <NoAccounts />
  if (user) return <Navigate to={safeNext(params.get('next'))} replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      setUser(await authApi.login(email.trim(), password))
      navigate(safeNext(params.get('next')), { replace: true })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <AuthCard title="Log in" intro={<>New here? <Link to="/signup" className="font-semibold text-navy underline">Create an account</Link>.</>}>
      <form onSubmit={submit} noValidate>
        <Problem message={error} />
        <Field id="email" label="Email" type="email" autoComplete="email" value={email} onChange={setEmail} />
        <Field id="password" label="Password" type="password" autoComplete="current-password" value={password} onChange={setPassword} />
        <Submit busy={busy} disabled={!email.trim() || !password}>Log in</Submit>
        <p className="mt-4 text-center text-sm"><Link to="/forgot" className="text-navy underline">Forgot your password?</Link></p>
      </form>
    </AuthCard>
  )
}

export function Signup() {
  const { hosted, user } = useApp()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [agreed, setAgreed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sent, setSent] = useState(false)
  if (!hosted) return <NoAccounts />
  if (user) return <Navigate to="/chat" replace />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (password.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters for the password.`)
      return
    }
    setBusy(true)
    setError(null)
    try {
      await authApi.signup(email.trim(), password)
      setSent(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <AuthCard title="Create an account" intro={<>Already have one? <Link to="/login" className="font-semibold text-navy underline">Log in</Link>.</>}>
      {sent ? (
        <Done>Check <strong>{email.trim()}</strong> for a link to confirm your address, then log in. The link expires in 24 hours.</Done>
      ) : (
        <form onSubmit={submit} noValidate>
          <Problem message={error} />
          <Field id="email" label="Email" type="email" autoComplete="email" value={email} onChange={setEmail} />
          <Field id="password" label="Password" type="password" autoComplete="new-password" value={password} onChange={setPassword} hint={`At least ${MIN_PASSWORD} characters.`} />
          <label className="mb-4 flex items-start gap-2 text-sm text-navy">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} className="mt-1" />
            <span>I accept the <Link to="/terms" className="underline">terms</Link> and the <Link to="/acceptable-use" className="underline">acceptable-use policy</Link>, and have read the <Link to="/privacy" className="underline">privacy notice</Link>.</span>
          </label>
          <Submit busy={busy} disabled={!email.trim() || !password || !agreed}>Create account</Submit>
        </form>
      )}
    </AuthCard>
  )
}

export function ForgotPassword() {
  const { hosted } = useApp()
  const [email, setEmail] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [sent, setSent] = useState(false)
  if (!hosted) return <NoAccounts />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await authApi.forgot(email.trim())
      setSent(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <AuthCard title="Forgot your password?" intro="Enter your email and we will send a link to choose a new one.">
      {sent ? (
        <Done>If <strong>{email.trim()}</strong> has an account, a reset link is on its way. It works once and expires in an hour.</Done>
      ) : (
        <form onSubmit={submit} noValidate>
          <Problem message={error} />
          <Field id="email" label="Email" type="email" autoComplete="email" value={email} onChange={setEmail} />
          <Submit busy={busy} disabled={!email.trim()}>Send reset link</Submit>
          <p className="mt-4 text-center text-sm"><Link to="/login" className="text-navy underline">Back to log in</Link></p>
        </form>
      )}
    </AuthCard>
  )
}

export function ResetPassword() {
  const { hosted } = useApp()
  const [params] = useSearchParams()
  const token = params.get('token') ?? ''
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [done, setDone] = useState(false)
  if (!hosted) return <NoAccounts />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (password.length < MIN_PASSWORD) {
      setError(`Use at least ${MIN_PASSWORD} characters for the password.`)
      return
    }
    setBusy(true)
    setError(null)
    try {
      await authApi.reset(token, password)
      setDone(true)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }
  return (
    <AuthCard title="Choose a new password">
      {done ? (
        <Done>Your password was changed and every other session was signed out. <Link to="/login" className="font-semibold underline">Log in</Link> with the new one.</Done>
      ) : !token ? (
        <Problem message="This page needs the link from the reset email." />
      ) : (
        <form onSubmit={submit} noValidate>
          <Problem message={error} />
          <Field id="password" label="New password" type="password" autoComplete="new-password" value={password} onChange={setPassword} hint={`At least ${MIN_PASSWORD} characters.`} />
          <Submit busy={busy} disabled={!password}>Change password</Submit>
        </form>
      )}
    </AuthCard>
  )
}

export function VerifyEmail() {
  const { hosted, user, setUser } = useApp()
  const [params] = useSearchParams()
  const token = params.get('token') ?? ''
  const [state, setState] = useState<'working' | 'done' | 'failed'>(token ? 'working' : 'failed')
  const [error, setError] = useState<string | null>(token ? null : 'This page needs the link from the confirmation email.')
  const started = useRef(false)

  useEffect(() => {
    // Tokens are single-use, so make sure a re-render (or StrictMode) never sends one twice.
    if (!hosted || !token || started.current) return
    started.current = true
    authApi
      .verify(token)
      .then(async () => {
        setState('done')
        const current = await authApi.me().catch(() => null)
        if (current) setUser(current)
      })
      .catch((err: Error) => {
        setState('failed')
        setError(err.message)
      })
  }, [hosted, token, setUser])

  if (!hosted) return <NoAccounts />
  return (
    <AuthCard title="Confirm your email">
      {state === 'working' ? <p className="text-sm text-navy/70" role="status">Confirming…</p> : null}
      {state === 'done' ? (
        <Done>Your address is confirmed. {user ? <Link to="/chat" className="font-semibold underline">Start chatting</Link> : <Link to="/login" className="font-semibold underline">Log in</Link>}.</Done>
      ) : null}
      {state === 'failed' ? <Problem message={error} /> : null}
    </AuthCard>
  )
}

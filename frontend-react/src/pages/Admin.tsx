import { motion } from 'framer-motion'
import { RefreshCw, Search, Trash2 } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { Card, ErrorNote, Page, PageHeader, SkeletonCards } from '../components/ui'
import { adminApi, type AdminUser } from '../lib/account'
import { GATE_LABELS, chatApi, type GateName } from '../lib/chat'
import { useApp } from '../lib/context'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <Card tilt={false}>
      <p className="text-sm text-navy/70">{label}</p>
      <p className="num mt-2 text-[28px] font-medium leading-none text-ink">{value}</p>
      {note ? <p className="mt-1 font-mono text-xs text-navy/70">{note}</p> : null}
    </Card>
  )
}

function Shares({ title, data }: { title: string; data: Record<string, number> }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0)
  return (
    <Card tilt={false}>
      <h2 className="mb-3 text-lg font-semibold text-ink">{title}</h2>
      {total ? (
        <ul className="space-y-2">
          {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([name, n]) => (
            <li key={name} className="text-sm">
              <div className="flex justify-between text-navy">
                <span>{GATE_LABELS[name as GateName] ?? name}</span>
                <span className="font-mono text-xs">{n} · {Math.round((n / total) * 100)}%</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-navy/10">
                <div className="h-1.5 rounded-full bg-navy" style={{ width: `${(n / total) * 100}%` }} />
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-navy/70">No turns yet.</p>
      )}
    </Card>
  )
}

/** Hosted mode: find a person and set their plan after their payment arrives. */
function PeoplePanel() {
  const [query, setQuery] = useState('')
  const [users, setUsers] = useState<AdminUser[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [days, setDays] = useState(31)

  const load = useCallback((q: string) => {
    adminApi.users(q).then((d) => { setUsers(d.users); setError(null) }, (err: Error) => setError(err.message))
  }, [])
  useEffect(() => load(''), [load])

  const setPlan = async (user: AdminUser, plan: string) => {
    try {
      const { user: updated } = await adminApi.setPlan(user.id, plan, days)
      setUsers((list) => list?.map((u) => (u.id === updated.id ? updated : u)) ?? null)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <Card tilt={false} className="mt-4">
      <h2 className="mb-1 text-lg font-semibold text-ink">People and plans</h2>
      <p className="mb-3 text-xs text-navy/70">
        When a payment arrives, find the person by the email on the payment and give them Pro. Paying again extends from the end
        of the current period; when it ends they are on Free again automatically.
      </p>
      <form className="mb-3 flex flex-wrap items-center gap-2" onSubmit={(e) => { e.preventDefault(); load(query) }}>
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search by email" aria-label="Search people by email" className="min-w-0 flex-1 rounded-lg border border-navy/15 bg-ice px-3 py-1.5 text-sm" />
        <button type="submit" className="inline-flex items-center gap-1.5 rounded-md border border-navy/15 px-3 py-1.5 text-xs text-navy hover:bg-navy/5"><Search className="h-3.5 w-3.5" /> Search</button>
        <label className="flex items-center gap-1.5 text-xs text-navy/70">
          Days of Pro
          <input type="number" min={1} max={3660} value={days} onChange={(e) => setDays(Math.max(1, Number(e.target.value) || 31))} className="w-20 rounded-md border border-navy/15 bg-ice px-2 py-1 font-mono text-xs" />
        </label>
      </form>
      {error ? <ErrorNote message={error} /> : null}
      {users ? (
        <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Recent turns">
          <table className="w-full min-w-[640px] font-mono text-xs">
            <thead className="text-navy/70">
              <tr><th className="text-left font-normal">Email</th><th className="text-left font-normal">Plan</th><th className="text-left font-normal">Paid until</th><th className="text-right font-normal">Today</th><th className="text-right font-normal" /></tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} className="border-t border-navy/10 text-navy">
                  <td className="py-1.5">{u.email}{u.verified ? '' : ' (unconfirmed)'}{u.is_admin ? ' · admin' : ''}</td>
                  <td>{u.effective_plan}{u.plan !== u.effective_plan ? ` (${u.plan} lapsed)` : ''}</td>
                  <td>{u.paid_until ? new Date(u.paid_until * 1000).toLocaleDateString() : '—'}</td>
                  <td className="text-right">{u.today.messages} · {u.today.tokens.toLocaleString()} tok</td>
                  <td className="space-x-1 text-right">
                    <button onClick={() => setPlan(u, 'pro')} className="rounded-md bg-navy px-2.5 py-0.5 text-ice">Pro +{days} d</button>
                    {u.plan !== 'free' ? <button onClick={() => setPlan(u, 'free')} className="rounded-md border border-wine/40 px-2.5 py-0.5 text-wine">Free</button> : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {!users.length ? <p className="py-2 text-sm text-navy/70">Nobody matches.</p> : null}
        </div>
      ) : null}
    </Card>
  )
}

export default function Admin() {
  const { hosted } = useApp()
  const stats = useApi(() => chatApi.stats(24))
  const status = useApi(chatApi.status)
  const gates = useApi(chatApi.gates)
  const [notice, setNotice] = useState<string | null>(null)

  const reload = () => {
    stats.reload()
    status.reload()
    gates.reload()
  }
  const clearCache = async () => {
    try {
      const { cleared } = await chatApi.clearCache()
      setNotice(`Cleared ${cleared} cached answer${cleared === 1 ? '' : 's'}.`)
      stats.reload()
    } catch (err) {
      setNotice((err as Error).message)
    }
  }

  const s = stats.data
  return (
    <Page>
      <PageHeader kicker="Operations" title="The admin view">
        Usage, cost and health of the chat assistant over the last 24 hours, with the state of every provider key. Keys are
        only ever shown by their last four characters.
      </PageHeader>

      <motion.div variants={rise} className="mb-4 flex flex-wrap gap-2">
        <button onClick={reload} className="inline-flex items-center gap-1.5 rounded-md border border-navy/15 px-3 py-1.5 text-xs text-navy hover:bg-navy/5">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
        <button onClick={clearCache} className="inline-flex items-center gap-1.5 rounded-md border border-wine/40 px-3 py-1.5 text-xs text-wine hover:bg-wine/10">
          <Trash2 className="h-3.5 w-3.5" /> Clear answer cache
        </button>
        {notice ? <span className="self-center text-xs text-navy/70" role="status">{notice}</span> : null}
      </motion.div>

      {stats.error ? <ErrorNote message={stats.error} /> : null}
      {stats.loading && !s ? <SkeletonCards count={4} height="h-14" /> : null}
      {s ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Chat turns" value={s.turns.toLocaleString()} note={`${s.cached} from the cache (${Math.round(s.cache_hit_rate * 100)}%)`} />
            <Stat label="Tokens spent" value={s.total_tokens.toLocaleString()} note={`≈ ${Math.round(s.avg_tokens_per_turn).toLocaleString()} per live turn`} />
            <Stat label="Latency" value={`${(s.latency_ms.p50 / 1000).toFixed(1)} s`} note={`p95 ${(s.latency_ms.p95 / 1000).toFixed(1)} s`} />
            <Stat label="Problems" value={`${s.errors + s.blocked}`} note={`${s.errors} failed · ${s.blocked} blocked`} />
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-3">
            <Shares title="Routed by" data={s.gates} />
            <Shares title="Decision" data={s.decisions} />
            <Shares title="Tokens by stage" data={s.tokens} />
          </div>
        </>
      ) : null}

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card tilt={false}>
          <h2 className="mb-3 text-lg font-semibold text-ink">Providers and keys</h2>
          {status.data ? (
            <>
              <p className="mb-2 text-xs text-navy/70">Order tried: {status.data.providers.join(' → ')}</p>
              <ul className="space-y-1.5 font-mono text-xs">
                {Object.entries(status.data.keys).flatMap(([provider, keys]) =>
                  keys.length
                    ? keys.map((k, i) => (
                        <li key={`${provider}${i}`} className="flex flex-wrap items-center gap-2">
                          <span className={`h-2 w-2 rounded-full ${k.usable ? 'bg-cyan' : k.invalid ? 'bg-wine' : 'bg-tangerine'}`} />
                          <span className="font-semibold text-ink">{provider}</span>
                          <span className="text-navy">{k.key}</span>
                          <span className="text-navy/70">{k.usable ? 'ready' : k.invalid ? 'rejected' : `cooling ${k.cooldown_seconds}s`} · {k.uses} calls</span>
                        </li>
                      ))
                    : [<li key={provider} className="text-navy/70">{provider}: no key</li>],
                )}
              </ul>
              <h3 className="mt-4 mb-1 text-sm font-semibold text-navy">Models by role</h3>
              <ul className="grid grid-cols-2 gap-x-3 font-mono text-[11px] text-navy/80">
                {Object.entries(status.data.models).map(([role, model]) => (
                  <li key={role} className="truncate" title={model}><span className="text-navy/70">{role}</span> {model}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-sm text-navy/70">{status.error ?? 'Loading…'}</p>
          )}
        </Card>

        <Card tilt={false}>
          <h2 className="mb-3 text-lg font-semibold text-ink">Gates</h2>
          {gates.data ? (
            <table className="w-full font-mono text-xs">
              <thead className="text-navy/70">
                <tr><th className="text-left font-normal">Gate</th><th className="text-right font-normal">≈ tokens</th><th className="text-right font-normal">accuracy</th><th className="text-right font-normal">ratings</th></tr>
              </thead>
              <tbody>
                {gates.data.gates.map((g) => (
                  <tr key={g.gate} className={g.chosen ? 'font-semibold text-ink' : 'text-navy'}>
                    <td>{GATE_LABELS[g.gate] ?? g.gate}{g.chosen ? ' (auto)' : ''}</td>
                    <td className="text-right">{Math.round(g.expected_tokens).toLocaleString()}</td>
                    <td className="text-right">{Math.round(g.accuracy * 100)}%</td>
                    <td className="text-right">{g.up}↑ {g.down}↓</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-sm text-navy/70">{gates.error ?? 'Loading…'}</p>
          )}
        </Card>
      </div>

      {s?.recent.length ? (
        <Card tilt={false} className="mt-4">
          <h2 className="mb-3 text-lg font-semibold text-ink">Recent turns</h2>
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="People and plans">
            <table className="w-full min-w-[560px] font-mono text-xs">
              <thead className="text-navy/70">
                <tr><th className="text-left font-normal">When</th><th className="text-left font-normal">Type</th><th className="text-left font-normal">Gate</th><th className="text-left font-normal">Decision</th><th className="text-right font-normal">Tokens</th><th className="text-right font-normal">Time</th></tr>
              </thead>
              <tbody>
                {s.recent.map((t, i) => (
                  <tr key={i} className="border-t border-navy/10 text-navy">
                    <td className="py-1">{new Date(t.ts * 1000).toLocaleTimeString()}</td>
                    <td>{t.task_type}</td>
                    <td>{GATE_LABELS[t.gate as GateName] ?? t.gate}{t.cached ? ' · cached' : ''}</td>
                    <td>{t.decision}</td>
                    <td className="text-right">{(t.total_tokens ?? 0).toLocaleString()}</td>
                    <td className="text-right">{((t.latency_ms ?? 0) / 1000).toFixed(1)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}
      {hosted ? <PeoplePanel /> : null}
    </Page>
  )
}

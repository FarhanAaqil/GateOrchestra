import { motion } from 'framer-motion'
import { RefreshCw, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Card, ErrorNote, Page, PageHeader, SkeletonCards } from '../components/ui'
import { GATE_LABELS, chatApi, type GateName } from '../lib/chat'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <Card tilt={false}>
      <p className="text-sm text-navy/60">{label}</p>
      <p className="mt-2 font-mono text-3xl font-semibold text-ink">{value}</p>
      {note ? <p className="mt-1 font-mono text-xs text-navy/60">{note}</p> : null}
    </Card>
  )
}

function Shares({ title, data }: { title: string; data: Record<string, number> }) {
  const total = Object.values(data).reduce((a, b) => a + b, 0)
  return (
    <Card tilt={false}>
      <h2 className="mb-3 text-lg font-bold text-ink">{title}</h2>
      {total ? (
        <ul className="space-y-2">
          {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([name, n]) => (
            <li key={name} className="text-sm">
              <div className="flex justify-between text-navy">
                <span>{GATE_LABELS[name as GateName] ?? name}</span>
                <span className="font-mono text-xs">{n} · {Math.round((n / total) * 100)}%</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-sky/40">
                <div className="h-1.5 rounded-full bg-navy" style={{ width: `${(n / total) * 100}%` }} />
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-navy/60">No turns yet.</p>
      )}
    </Card>
  )
}

export default function Admin() {
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
      <PageHeader number="09" kicker="Operations" title={<>The <span className="mark-lemon">admin</span> view</>}>
        Usage, cost and health of the chat assistant over the last 24 hours, with the state of every provider key. Keys are
        only ever shown by their last four characters.
      </PageHeader>

      <motion.div variants={rise} className="mb-4 flex flex-wrap gap-2">
        <button onClick={reload} className="inline-flex items-center gap-1.5 rounded-full border border-sky px-3 py-1.5 text-xs text-navy hover:bg-soft">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
        <button onClick={clearCache} className="inline-flex items-center gap-1.5 rounded-full border border-wine/40 px-3 py-1.5 text-xs text-wine hover:bg-wine/10">
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
          <h2 className="mb-3 text-lg font-bold text-ink">Providers and keys</h2>
          {status.data ? (
            <>
              <p className="mb-2 text-xs text-navy/60">Order tried: {status.data.providers.join(' → ')}</p>
              <ul className="space-y-1.5 font-mono text-xs">
                {Object.entries(status.data.keys).flatMap(([provider, keys]) =>
                  keys.length
                    ? keys.map((k, i) => (
                        <li key={`${provider}${i}`} className="flex flex-wrap items-center gap-2">
                          <span className={`h-2 w-2 rounded-full ${k.usable ? 'bg-cyan' : k.invalid ? 'bg-wine' : 'bg-tangerine'}`} />
                          <span className="font-semibold text-ink">{provider}</span>
                          <span className="text-navy">{k.key}</span>
                          <span className="text-navy/60">{k.usable ? 'ready' : k.invalid ? 'rejected' : `cooling ${k.cooldown_seconds}s`} · {k.uses} calls</span>
                        </li>
                      ))
                    : [<li key={provider} className="text-navy/50">{provider}: no key</li>],
                )}
              </ul>
              <h3 className="mt-4 mb-1 text-sm font-semibold text-navy">Models by role</h3>
              <ul className="grid grid-cols-2 gap-x-3 font-mono text-[11px] text-navy/80">
                {Object.entries(status.data.models).map(([role, model]) => (
                  <li key={role} className="truncate" title={model}><span className="text-navy/50">{role}</span> {model}</li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-sm text-navy/60">{status.error ?? 'Loading…'}</p>
          )}
        </Card>

        <Card tilt={false}>
          <h2 className="mb-3 text-lg font-bold text-ink">Gates</h2>
          {gates.data ? (
            <table className="w-full font-mono text-xs">
              <thead className="text-navy/60">
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
            <p className="text-sm text-navy/60">{gates.error ?? 'Loading…'}</p>
          )}
        </Card>
      </div>

      {s?.recent.length ? (
        <Card tilt={false} className="mt-4">
          <h2 className="mb-3 text-lg font-bold text-ink">Recent turns</h2>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] font-mono text-xs">
              <thead className="text-navy/60">
                <tr><th className="text-left font-normal">When</th><th className="text-left font-normal">Type</th><th className="text-left font-normal">Gate</th><th className="text-left font-normal">Decision</th><th className="text-right font-normal">Tokens</th><th className="text-right font-normal">Time</th></tr>
              </thead>
              <tbody>
                {s.recent.map((t, i) => (
                  <tr key={i} className="border-t border-sky/50 text-navy">
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
    </Page>
  )
}

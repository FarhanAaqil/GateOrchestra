import { motion } from 'framer-motion'
import { RefreshCw } from 'lucide-react'
import { Card, EmptyState, ErrorNote, Page, PageHeader, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { METHOD_LABELS, num, STRATEGY_LABELS } from '../lib/format'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

/** Probe (sky) and MAS (ember) share of one run's spend; wine marks MAS over its cap. */
function SpendBar({ probe, mas, over, max }: { probe: number; mas: number; over: boolean; max: number }) {
  const scale = (v: number) => `${(v / Math.max(max, 1)) * 100}%`
  return (
    <div className="flex h-2.5 w-36 overflow-hidden rounded-full bg-sky/30">
      <span className="h-full bg-sky" style={{ width: scale(probe) }} />
      <span className={`h-full ${over ? 'bg-wine' : 'bg-ember'}`} style={{ width: scale(mas) }} />
    </div>
  )
}

export default function TokenAccountant() {
  const history = useApi(api.history)
  const detail = useApi(api.resultsDetail)
  const runs = history.data?.history ?? []
  const benchmarkRows = detail.data?.per_task ?? []
  const totals = runs.reduce((acc, r) => ({ probe: acc.probe + (r.probe_tokens ?? 0), mas: acc.mas + (r.mas_tokens ?? 0) }), { probe: 0, mas: 0 })
  const maxSpend = Math.max(1, ...runs.map((r) => r.tokens_spent))

  return (
    <Page>
      <PageHeader number="07" kicker="Token accountant" title={<>Every token, <span className="mark-lemon">by path</span></>}>
        Spend is the provider's own prompt + completion count for every call. It is never clipped to the budget.
      </PageHeader>

      <div className="mb-6 grid gap-4 sm:grid-cols-3">
        <Card navy>
          <p className="text-sm text-sky">Runs this server session</p>
          <p className="mt-2 font-mono text-5xl font-semibold text-ice">{runs.length}</p>
        </Card>
        <Card>
          <p className="flex items-center gap-2 text-sm text-navy/60"><span className="h-2.5 w-2.5 rounded-full bg-sky" /> Probe tokens</p>
          <p className="mt-2 font-mono text-5xl font-semibold text-ink">{num(totals.probe)}</p>
        </Card>
        <Card>
          <p className="flex items-center gap-2 text-sm text-navy/60"><span className="h-2.5 w-2.5 rounded-full bg-ember" /> MAS tokens</p>
          <p className="mt-2 font-mono text-5xl font-semibold text-ember">{num(totals.mas)}</p>
        </Card>
      </div>

      <motion.div variants={rise} className="mb-3 flex items-center gap-2">
        <h2 className="text-2xl font-bold text-ink">Live runs</h2>
        <button onClick={history.reload} className="ml-auto inline-flex items-center gap-1.5 rounded-full border border-sky px-3 py-1.5 text-xs text-navy hover:bg-soft">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
      </motion.div>
      {history.loading ? <Spinner /> : null}
      {history.error ? <ErrorNote message={history.error} /> : null}
      {history.data && !runs.length ? <EmptyState title="No runs yet">Runs from Run a Task appear here (the API keeps the last 50 in memory).</EmptyState> : null}
      {runs.length ? (
        <motion.div variants={rise} className="card overflow-x-auto p-0">
          <table className="w-full min-w-[860px] text-left text-sm">
            <thead className="bg-navy font-mono text-xs uppercase tracking-wider text-sky">
              <tr>
                <th className="px-5 py-3.5">Task</th>
                <th className="px-5 py-3.5">Method</th>
                <th className="px-5 py-3.5">Path</th>
                <th className="px-5 py-3.5">Spend</th>
                <th className="px-5 py-3.5 text-right">Probe</th>
                <th className="px-5 py-3.5 text-right">MAS</th>
                <th className="px-5 py-3.5 text-right">Cap</th>
                <th className="px-5 py-3.5 text-right">Total</th>
                <th className="px-5 py-3.5">Outcome</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => {
                const cap = r.gate_decision?.token_budget_cap ?? null
                const over = cap != null && (r.mas_tokens ?? 0) > cap
                const path = r.mas_tokens ? `MAS${r.mas_strategy ? ` · ${STRATEGY_LABELS[r.mas_strategy]}` : ''}` : 'CoT-SC'
                return (
                  <tr key={`${r.task_id}-${i}`} className="border-t border-sky/60 hover:bg-ice">
                    <td className="px-5 py-3 font-mono text-xs text-ink">{r.task_id}</td>
                    <td className="px-5 py-3 text-ink">{METHOD_LABELS[r.method]}</td>
                    <td className="px-5 py-3">
                      <span className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] ${r.mas_tokens ? 'bg-ember text-soft' : 'bg-sky text-ink'}`}>{path}</span>
                    </td>
                    <td className="px-5 py-3"><SpendBar probe={r.probe_tokens ?? 0} mas={r.mas_tokens ?? 0} over={over} max={maxSpend} /></td>
                    <td className="px-5 py-3 text-right font-mono">{num(r.probe_tokens)}</td>
                    <td className={`px-5 py-3 text-right font-mono ${over ? 'font-semibold text-wine' : ''}`}>{num(r.mas_tokens)}</td>
                    <td className="px-5 py-3 text-right font-mono text-navy/60">{num(cap)}</td>
                    <td className="px-5 py-3 text-right font-mono font-semibold text-tangerine">{num(r.tokens_spent)}</td>
                    <td className="px-5 py-3 text-xs">
                      {r.simulated ? (
                        <span className="rounded-full border border-sky px-2 py-0.5 text-navy/60">simulated</span>
                      ) : r.is_correct === true ? (
                        <span className="rounded-full bg-navy px-2 py-0.5 text-cyan">correct</span>
                      ) : r.is_correct === false ? (
                        <span className="rounded-full bg-wine px-2 py-0.5 text-soft">incorrect</span>
                      ) : (
                        <span className="text-navy/60">ungraded</span>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </motion.div>
      ) : null}

      <motion.h2 variants={rise} className="mb-3 mt-10 text-2xl font-bold text-ink">Benchmark runs</motion.h2>
      {detail.loading ? (
        <Spinner />
      ) : benchmarkRows.length ? (
        <motion.div variants={rise} className="card max-h-[50vh] overflow-auto p-0">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead className="sticky top-0 bg-navy font-mono text-xs uppercase tracking-wider text-sky">
              <tr>
                <th className="px-5 py-3">Seed</th>
                <th className="px-5 py-3">Task</th>
                <th className="px-5 py-3">Method</th>
                <th className="px-5 py-3">Decision</th>
                <th className="px-5 py-3 text-right">Tokens</th>
                <th className="px-5 py-3">Correct</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkRows.map((r, i) => (
                <tr key={i} className="border-t border-sky/60">
                  <td className="px-5 py-2 font-mono text-xs text-navy/60">{r.seed}</td>
                  <td className="px-5 py-2 font-mono text-xs text-ink">{r.task_id}</td>
                  <td className="px-5 py-2 text-ink">{METHOD_LABELS[r.method]}</td>
                  <td className="px-5 py-2 font-mono text-xs">{r.decision ?? '—'}</td>
                  <td className="px-5 py-2 text-right font-mono text-tangerine">{num(r.tokens_spent)}</td>
                  <td className="px-5 py-2 text-xs">{r.is_correct ? 'yes' : 'no'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      ) : (
        <EmptyState title="No per-task benchmark records">
          The recorded results predate per-task logging. They appear here after{' '}
          <code className="rounded bg-lemon px-1 font-mono text-ink">python scripts/final_benchmark.py</code>.
        </EmptyState>
      )}
    </Page>
  )
}

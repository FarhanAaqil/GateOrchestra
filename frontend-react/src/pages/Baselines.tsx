import { AccuracyChart, TokensChart } from '../charts/BaselineChart'
import { Card, NoCitableResults, Page, PageHeader, Spinner, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import { isCitable, METHOD_COLORS, METHOD_LABELS, num, pct } from '../lib/format'
import type { ResultMethod } from '../lib/types'
import { useApi } from '../lib/useApi'

const METHODS: { method: ResultMethod; text: string }[] = [
  { method: 'CoT-SC-only', text: 'Always answer with the probe. The strong single-agent reference.' },
  { method: 'Always-MAS', text: 'Always escalate, ignoring the gate. The cost upper bound.' },
  { method: 'RandomGate', text: "Coin-flip escalation at the learned gate's validation escalation rate." },
  { method: 'RuleBasedGate', text: 'Hand-written thresholds on consistency, entities, depth and parallelism.' },
  { method: 'GateOrchestra', text: 'The learned gate with the k × probe-token escalation cap.' },
]

export default function Baselines() {
  const { data, loading, error } = useApi(api.summary)
  const rows = data?.aggregated ?? []
  const citable = isCitable(data?.status)
  const tests = data?.pooled_paired_tests ?? []

  return (
    <Page>
      <PageHeader number="05" kicker="Baselines" title={<>Five methods, <span className="mark-lemon">one paired</span> benchmark</>}>
        Every method answers the same test tasks with the same probe, under the same k × probe-token budget. Error bars are pooled
        95% Wilson intervals.
      </PageHeader>

      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {METHODS.map(({ method, text }) => (
          <Card key={method} navy={method === 'GateOrchestra'}>
            <span className="block h-2 w-10 rounded-full" style={{ background: METHOD_COLORS[method] }} />
            <p className={`mt-3 font-bold ${method === 'GateOrchestra' ? 'text-ice' : 'text-ink'}`}>{METHOD_LABELS[method]}</p>
            <p className={`mt-1 text-xs ${method === 'GateOrchestra' ? 'text-sky' : 'text-navy/65'}`}>{text}</p>
          </Card>
        ))}
      </div>

      {data ? <div className="mb-4"><StatusBadge status={data.status} /></div> : null}
      {loading ? <Spinner /> : null}
      {error ? <p className="text-wine">{error}</p> : null}

      {data && citable && rows.length ? (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card tilt={false}>
              <h2 className="mb-2 text-xl font-bold text-ink">Accuracy (95% Wilson CI)</h2>
              <AccuracyChart rows={rows} />
            </Card>
            <Card tilt={false}>
              <h2 className="mb-2 text-xl font-bold text-ink">Average tokens per task</h2>
              <TokensChart rows={rows} />
            </Card>
          </div>

          <Card tilt={false} className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead className="font-mono text-xs uppercase tracking-wider text-navy/60">
                <tr>
                  <th className="py-2">Method</th>
                  <th>Accuracy</th>
                  <th>95% CI</th>
                  <th>Avg tokens</th>
                  <th>Saved vs MAS</th>
                  <th>Escalates</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.method} className={`border-t border-sky/60 ${r.method === 'GateOrchestra' ? 'bg-lemon/50' : ''}`}>
                    <td className="py-3 font-semibold text-ink">{METHOD_LABELS[r.method]}</td>
                    <td className="font-mono">{pct(r.accuracy_mean)}</td>
                    <td className="font-mono text-navy/60">
                      {r.accuracy_pooled_ci95 ? `${r.accuracy_pooled_ci95[0].toFixed(1)}–${r.accuracy_pooled_ci95[1].toFixed(1)}%` : '—'}
                    </td>
                    <td className="font-mono text-tangerine">{num(r.avg_tokens_mean)}</td>
                    <td className="font-mono">{pct(r.token_savings_pct_mean)}</td>
                    <td className="font-mono">{r.escalate_rate_mean != null ? pct(r.escalate_rate_mean * 100) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>

          {tests.length ? (
            <Card tilt={false} className="mt-4">
              <h2 className="text-xl font-bold text-ink">Paired McNemar tests</h2>
              <p className="mb-3 text-xs text-navy/60">Seeds share test tasks, so pooled p-values are optimistic.</p>
              <div className="grid gap-2 sm:grid-cols-2">
                {tests.map((t) => (
                  <div key={`${t.method}-${t.reference}`} className="rounded-2xl border border-sky bg-ice px-3 py-2 font-mono text-xs text-ink">
                    {t.method} vs {t.reference}: {t.method_only_correct}–{t.reference_only_correct} discordant,{' '}
                    <span className={`rounded-full px-1.5 ${t.p_value < 0.05 ? 'bg-navy text-cyan' : 'text-navy/60'}`}>p = {t.p_value.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </Card>
          ) : null}
        </>
      ) : data ? (
        <NoCitableResults status={data.status} what="baseline comparison" />
      ) : null}
    </Page>
  )
}

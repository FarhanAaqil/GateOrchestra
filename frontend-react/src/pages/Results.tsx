import { useState } from 'react'
import AxisHeatmap from '../charts/AxisHeatmap'
import KSweepChart from '../charts/KSweepChart'
import ParetoChart from '../charts/ParetoChart'
import { Card, NoCitableResults, Page, PageHeader, SkeletonCards, StatusBadge, Term } from '../components/ui'
import { api } from '../lib/api'
import { isCitable, METHOD_LABELS } from '../lib/format'
import { FIELD } from '../lib/styles'
import type { ResultMethod } from '../lib/types'
import { useApi } from '../lib/useApi'

const METHODS: ResultMethod[] = ['GateOrchestra', 'RuleBasedGate', 'RandomGate', 'CoT-SC-only', 'Always-MAS']

export default function Results() {
  const [run, setRun] = useState('final')
  const runs = useApi(api.runs)
  const summary = useApi(() => api.summary(run), run)
  const detail = useApi(() => api.resultsDetail(run), run)
  const runList = runs.data?.runs ?? []
  const current = runList.find((r) => r.id === run)
  const [method, setMethod] = useState<ResultMethod>('GateOrchestra')
  const [metric, setMetric] = useState<'accuracy' | 'escalation_rate'>('accuracy')
  const status = summary.data?.status
  const citable = isCitable(status)

  return (
    <Page>
      <PageHeader number="06" kicker="Results" title={<>Accuracy, cost, and <span className="mark-lemon">where it escalates</span></>}>
        Every chart is computed from a recorded benchmark run. The main run is Qwen2.5-3B with two seeds; the larger models were
        run once each with fewer calibration repeats (section 8.9 of the research report).
      </PageHeader>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {runList.length > 1 ? (
          <label className="flex items-center gap-2 text-sm text-navy">
            Benchmark run
            <select value={run} onChange={(e) => setRun(e.target.value)} className="rounded-full border border-sky bg-soft px-3 py-1.5 text-sm text-ink">
              {runList.map((r) => (
                <option key={r.id} value={r.id}>{r.label}</option>
              ))}
            </select>
          </label>
        ) : null}
        {status ? <StatusBadge status={status} /> : null}
        {current && !current.main ? (
          <span className="text-xs text-navy/60">One seed only, so there is no run-to-run spread. Shown for comparison, not as the headline result.</span>
        ) : null}
      </div>
      {(summary.loading && !summary.data) || (detail.loading && !detail.data) ? <SkeletonCards count={3} height="h-48" /> : null}

      {summary.data && !citable ? <NoCitableResults status={status} what="results" /> : null}

      {citable && summary.data && detail.data ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card tilt={false} className="lg:col-span-2">
            <h2 className="text-xl font-bold text-ink">Accuracy vs average tokens</h2>
            <p className="mb-2 text-xs text-navy/60">Up and to the left is better. GateOrchestra is the ember star.</p>
            <ParetoChart rows={summary.data.aggregated} />
          </Card>

          <Card tilt={false}>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h2 className="mr-auto text-xl font-bold text-ink">Depth × Parallel</h2>
              <select value={method} onChange={(e) => setMethod(e.target.value as ResultMethod)} className={`${FIELD} w-auto py-1.5 text-xs`}>
                {METHODS.map((m) => (
                  <option key={m} value={m}>
                    {METHOD_LABELS[m]}
                  </option>
                ))}
              </select>
              <select value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} className={`${FIELD} w-auto py-1.5 text-xs`}>
                <option value="accuracy">Accuracy</option>
                <option value="escalation_rate">Escalation rate</option>
              </select>
            </div>
            <AxisHeatmap cells={detail.data.axis_cells} method={method} metric={metric} />
          </Card>

          <Card tilt={false}>
            <h2 className="text-xl font-bold text-ink">Escalation precision / recall</h2>
            <p className="mb-1 text-xs text-navy/60">Against the oracle: a task needs the agent team (MAS) when MAS is right and the quick check (CoT-SC) is wrong.</p>
            <p className="mb-3 text-xs text-navy/60">
              <Term term="precision">Precision</Term>: of the tasks sent to the agent team, how many needed it.{' '}
              <Term term="recall">Recall</Term>: of the tasks that needed it, how many were sent.
            </p>
            <div className="space-y-3">
              {detail.data.escalation_quality.map((q) => (
                <div key={q.method} className={`rounded-2xl p-4 ${q.method === 'GateOrchestra' ? 'bg-navy text-ice' : 'border border-sky bg-ice text-ink'}`}>
                  <p className="font-bold">{METHOD_LABELS[q.method]}</p>
                  <div className="mt-1 grid grid-cols-2 gap-2 font-mono text-sm">
                    <span>
                      <Term term="precision">precision</Term> <span className={`font-semibold ${q.method === 'GateOrchestra' ? 'text-ember' : 'text-wine'}`}>{q.precision == null ? '—' : `${(q.precision * 100).toFixed(0)}%`}</span>
                    </span>
                    <span>
                      <Term term="recall">recall</Term> <span className={`font-semibold ${q.method === 'GateOrchestra' ? 'text-ember' : 'text-wine'}`}>{q.recall == null ? '—' : `${(q.recall * 100).toFixed(0)}%`}</span>
                    </span>
                  </div>
                  <p className="mt-1 font-mono text-[11px] opacity-70">
                    <Term term="tp">TP</Term> {q.true_positive} · <Term term="fp">FP</Term> {q.false_positive} · <Term term="fn">FN</Term> {q.false_negative}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card tilt={false} className="lg:col-span-2">
            <h2 className="text-xl font-bold text-ink">k sweep (validation)</h2>
            <p className="mb-2 text-xs text-navy/60">The selected classifier and <Term term="tauAcc">τ_acc</Term> per seed, averaged across seeds. <Term term="k">k</Term> is the agent team's token budget.</p>
            {detail.data.k_sweep.length ? (
              <KSweepChart points={detail.data.k_sweep} />
            ) : (
              <p className="text-sm text-navy/70">These results predate calibration logging; re-run the benchmark to record the sweep.</p>
            )}
          </Card>
        </div>
      ) : null}
    </Page>
  )
}

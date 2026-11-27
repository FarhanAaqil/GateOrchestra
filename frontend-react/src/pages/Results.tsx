import { useState } from 'react'
import AxisHeatmap from '../charts/AxisHeatmap'
import KSweepChart from '../charts/KSweepChart'
import ParetoChart from '../charts/ParetoChart'
import ThresholdComparison from '../components/ThresholdComparison'
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
      <PageHeader kicker="Results" title="Accuracy, cost, and where it escalates">
        Every chart is computed from a recorded benchmark run. The main run is Qwen2.5-3B with two seeds; the larger models were
        run once each with fewer calibration repeats (section 8.9 of the research report).
      </PageHeader>
      <div className="mb-4 flex flex-wrap items-center gap-3">
        {runList.length > 1 ? (
          <label className="flex items-center gap-2 text-sm text-muted">
            Benchmark run
            <select value={run} onChange={(e) => setRun(e.target.value)} className="rounded-lg border border-line bg-paper px-3 py-1.5 text-sm text-ink outline-none focus:border-forest/40">
              {runList.map((r) => (
                <option key={r.id} value={r.id}>{r.label}</option>
              ))}
            </select>
          </label>
        ) : null}
        {status ? <StatusBadge status={status} /> : null}
        {current && !current.main ? (
          <span className="text-xs text-muted">One seed only, so there is no run-to-run spread. Shown for comparison, not as the headline result.</span>
        ) : null}
      </div>
      {(summary.loading && !summary.data) || (detail.loading && !detail.data) ? <SkeletonCards count={3} height="h-48" /> : null}

      {summary.data && !citable ? <NoCitableResults status={status} what="results" /> : null}

      {citable && summary.data && detail.data ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card tilt={false} className="lg:col-span-2">
            <h2 className="text-lg font-semibold text-ink">Accuracy vs average tokens</h2>
            <p className="mb-2 text-xs text-muted">Up and to the left is better. GateOrchestra is the burgundy star.</p>
            <ParetoChart rows={summary.data.aggregated} />
          </Card>

          <Card tilt={false}>
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <h2 className="mr-auto text-lg font-semibold text-ink">Depth × Parallel</h2>
              <select aria-label="Method shown in the heatmap" value={method} onChange={(e) => setMethod(e.target.value as ResultMethod)} className={`${FIELD} w-auto py-1.5 text-xs`}>
                {METHODS.map((m) => (
                  <option key={m} value={m}>
                    {METHOD_LABELS[m]}
                  </option>
                ))}
              </select>
              <select aria-label="Metric shown in the heatmap" value={metric} onChange={(e) => setMetric(e.target.value as typeof metric)} className={`${FIELD} w-auto py-1.5 text-xs`}>
                <option value="accuracy">Accuracy</option>
                <option value="escalation_rate">Escalation rate</option>
              </select>
            </div>
            <AxisHeatmap cells={detail.data.axis_cells} method={method} metric={metric} />
          </Card>

          <Card tilt={false}>
            <h2 className="text-lg font-semibold text-ink">Escalation precision / recall</h2>
            <p className="mb-1 text-xs text-muted">Against the oracle: a task needs the agent team (MAS) when MAS is right and the quick check (CoT-SC) is wrong.</p>
            <p className="mb-3 text-xs text-muted">
              <Term term="precision">Precision</Term>: of the tasks sent to the agent team, how many needed it.{' '}
              <Term term="recall">Recall</Term>: of the tasks that needed it, how many were sent.
            </p>
            <div className="space-y-3">
              {detail.data.escalation_quality.map((q) => (
                <div key={q.method} className={`rounded-lg border p-4 text-ink ${q.method === 'GateOrchestra' ? 'border-clay/40 bg-clay/[0.05]' : 'border-line bg-canvas/60'}`}>
                  <p className="font-semibold">{METHOD_LABELS[q.method]}</p>
                  <div className="mt-1 grid grid-cols-2 gap-2 text-sm text-muted">
                    <span>
                      <Term term="precision">precision</Term> <span className="num font-semibold text-ink">{q.precision == null ? '—' : `${(q.precision * 100).toFixed(0)}%`}</span>
                    </span>
                    <span>
                      <Term term="recall">recall</Term> <span className="num font-semibold text-ink">{q.recall == null ? '—' : `${(q.recall * 100).toFixed(0)}%`}</span>
                    </span>
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-muted">
                    <Term term="tp">TP</Term> {q.true_positive} · <Term term="fp">FP</Term> {q.false_positive} · <Term term="fn">FN</Term> {q.false_negative}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          {run === 'final' ? <ThresholdComparison /> : null}

          <Card tilt={false} className="lg:col-span-2">
            <h2 className="text-lg font-semibold text-ink">k sweep (validation)</h2>
            <p className="mb-2 text-xs text-muted">The selected classifier and <Term term="tauAcc">τ_acc</Term> per seed, averaged across seeds. <Term term="k">k</Term> is the agent team's token budget.</p>
            {detail.data.k_sweep.length ? (
              <KSweepChart points={detail.data.k_sweep} />
            ) : (
              <p className="text-sm text-muted">These results predate calibration logging; re-run the benchmark to record the sweep.</p>
            )}
          </Card>
        </div>
      ) : null}
    </Page>
  )
}

import { Link } from 'react-router-dom'
import ConfidenceGauge from '../charts/ConfidenceGauge'
import ImportanceChart from '../charts/ImportanceChart'
import { Card, EmptyState, Page, PageHeader, Skeleton } from '../components/ui'
import { api } from '../lib/api'
import { useApp } from '../lib/context'
import { FEATURE_LABELS } from '../lib/format'
import { useApi } from '../lib/useApi'

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-forest/[0.06] py-2 last:border-0">
      <dt className="text-sm text-forest/70">{label}</dt>
      <dd className="num text-sm text-ink">{value}</dd>
    </div>
  )
}

/** Where to try a gated run: Run a Task for operators, the chat for everyone else. */
function TryLink({ canRun }: { canRun: boolean }) {
  const cls = 'font-medium text-ink underline decoration-forest/30 underline-offset-2'
  return canRun ? <Link to="/run" className={cls}>Run a Task</Link> : <Link to="/chat" className={cls}>the chat</Link>
}

export default function GateInspector() {
  const { models, lastRun, threshold, setThreshold, isAdmin } = useApp()
  const importance = useApi(api.gateImportance)
  const detail = useApi(api.resultsDetail)
  const gate = models?.trained_gate ?? null
  const decision = lastRun?.gate_decision ?? null
  const learned = decision?.gate_type === 'learned'
  const escalateProbability = decision ? (decision.decision === 'ESCALATE' ? decision.confidence : 1 - decision.confidence) : null
  const tauForGate = detail.data?.gates.find((g) => g.seed === gate?.seed)?.tau_acc
  const wouldEscalate = learned && escalateProbability != null && threshold != null ? escalateProbability >= threshold : null

  return (
    <Page>
      <PageHeader kicker="Gate inspector" title="Inside the routing decision">
        The gate reads eight pre-execution features and outputs the probability that multi-agent orchestration is worth its cost on
        a task.
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <p className="eyebrow">Loaded gate</p>
          {gate ? (
            <dl className="mt-3">
              <Row label="Classifier" value={gate.classifier} />
              <Row label="Benchmark seed" value={gate.seed ?? '—'} />
              <Row label="τ_acc (labels)" value={tauForGate ?? '—'} />
              <Row label="Features via" value={gate.build_info?.feature_backend ?? '—'} />
              <Row label="scikit-learn" value={gate.build_info?.sklearn_version ?? '—'} />
              {gate.degenerate ? (
                <p className="mt-3 rounded-lg border border-amber/40 bg-amber/10 px-3 py-2 text-sm text-ink">
                  Degenerate: every training label was the same class, so this gate always gives one answer.
                </p>
              ) : null}
            </dl>
          ) : (
            <p className="mt-3 text-sm text-forest/70">
              No trained gate yet. The benchmark's calibration step trains and saves one; until then the GateOrchestra method is disabled.
            </p>
          )}
        </Card>

        <Card>
          <p className="eyebrow">Last decision</p>
          {decision && escalateProbability != null ? (
            <div className="mt-2 flex flex-col items-center">
              <ConfidenceGauge escalateProbability={escalateProbability} threshold={learned ? threshold : null} />
              <p className="text-sm text-ink">
                <span className={`chip ${decision.decision === 'STOP' ? 'border-sage! bg-sage/25!' : 'border-clay/50! bg-clay/10!'}`}>{decision.decision}</span>
                <span className="ml-2 text-forest/70">{decision.gate_type} gate</span>
              </p>
              <p className="mt-2 text-center text-xs text-forest/70">
                {learned
                  ? `P(escalate) = ${(escalateProbability * 100).toFixed(1)}%`
                  : decision.gate_type === 'random'
                    ? 'Random gate: a coin flip, not a probability.'
                    : 'Rule-based gate: a deterministic rule, not a probability.'}
              </p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-forest/70">
              {isAdmin ? (
                <>Run a gated method on <TryLink canRun /> to inspect its decision here.</>
              ) : (
                <>Decisions from operator test runs appear here. Every answer in <TryLink canRun={false} /> also shows how it was routed.</>
              )}
            </p>
          )}
        </Card>

        <Card>
          <p className="eyebrow">Escalation threshold</p>
          <p className="mt-2 text-xs leading-relaxed text-forest/70">
            Applies to GateOrchestra runs from this browser; otherwise the classifier's own decision (about 0.5) is used.
          </p>
          <label className="mt-4 flex items-center gap-2 text-sm font-medium text-ink">
            <input type="checkbox" checked={threshold != null} onChange={(e) => setThreshold(e.target.checked ? 0.5 : null)} className="h-4 w-4 accent-forest" />
            Override threshold
          </label>
          <input
            type="range"
            min={0.05}
            max={0.95}
            step={0.01}
            value={threshold ?? 0.5}
            disabled={threshold == null}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="mt-3 w-full accent-clay disabled:opacity-40"
            aria-label="Escalation threshold"
          />
          <p className="num mt-1 text-2xl font-medium text-ink">{threshold == null ? 'Default' : threshold.toFixed(2)}</p>
          {wouldEscalate != null ? (
            <p className="mt-2 text-sm text-ink">
              At this threshold the last task would{' '}
              <span className={`chip ${wouldEscalate ? 'border-clay/50! bg-clay/10!' : 'border-sage! bg-sage/25!'}`}>{wouldEscalate ? 'ESCALATE' : 'STOP'}</span>
            </p>
          ) : null}
          <p className="mt-3 text-xs leading-relaxed text-forest/70">
            τ_acc is different: it is the accuracy gap used to label training tasks, so changing it means retraining the gate.
          </p>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <h2 className="mb-3 text-lg font-semibold text-ink">Feature importance</h2>
          {importance.loading ? (
            <Skeleton className="h-48" />
          ) : importance.data?.available ? (
            <ImportanceChart rows={importance.data.importances} />
          ) : (
            <div className="rounded-lg border border-dashed border-forest/20 px-4 py-5 text-sm text-forest/70">
              <p>{importance.data?.reason ?? importance.error ?? 'Unavailable.'}</p>
              <p className="mt-2 text-xs text-forest/70">
                Some gate types cannot report per-feature importance. Section 8.4 of the research report tests which features the gate
                depends on by removing them one at a time, and section 8.10 compares the gate with a threshold on its main feature.
              </p>
            </div>
          )}
        </Card>
        <Card>
          <h2 className="mb-3 text-lg font-semibold text-ink">Features of the last gated run</h2>
          {lastRun?.features ? (
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-forest/10 bg-forest/10">
              {Object.entries(lastRun.features)
                .filter(([key]) => key !== 'task_id')
                .map(([key, value]) => (
                  <div key={key} className="bg-paper px-3 py-2.5">
                    <dt className="text-xs text-forest/70">{FEATURE_LABELS[key] ?? key}</dt>
                    <dd className="num mt-0.5 text-base text-ink">
                      {typeof value === 'boolean' ? (value ? 'yes' : 'no') : typeof value === 'number' ? Number(value.toFixed(2)) : '—'}
                    </dd>
                  </div>
                ))}
            </dl>
          ) : (
            <EmptyState title="No gated run yet">
              {isAdmin ? (
                <>Try a question on <TryLink canRun /> with GateOrchestra, the rule-based gate or the random gate to see the eight features the gate looks at.</>
              ) : (
                <>An operator's test run fills this in with the eight features the gate looks at.</>
              )}
            </EmptyState>
          )}
        </Card>
      </div>
    </Page>
  )
}

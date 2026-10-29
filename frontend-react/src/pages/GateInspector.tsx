import { Link } from 'react-router-dom'
import ConfidenceGauge from '../charts/ConfidenceGauge'
import ImportanceChart from '../charts/ImportanceChart'
import { Card, EmptyState, Page, PageHeader, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { useApp } from '../lib/context'
import { FEATURE_LABELS } from '../lib/format'
import { useApi } from '../lib/useApi'

export default function GateInspector() {
  const { models, lastRun, threshold, setThreshold } = useApp()
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
      <PageHeader number="03" kicker="Gate inspector" title={<>Inside the <span className="mark-lemon">routing decision</span></>}>
        The gate sees eight pre-execution features and outputs a probability that multi-agent orchestration is worth its cost on
        this task.
      </PageHeader>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card navy>
          <p className="font-mono text-xs uppercase tracking-widest text-sky">Loaded gate</p>
          {gate ? (
            <dl className="mt-4 grid grid-cols-2 gap-y-2 font-mono text-sm">
              <dt className="text-sky/70">Classifier</dt>
              <dd className="text-cyan">{gate.classifier}</dd>
              <dt className="text-sky/70">Benchmark seed</dt>
              <dd className="text-ice">{gate.seed ?? '—'}</dd>
              <dt className="text-sky/70">τ_acc (labels)</dt>
              <dd className="text-ice">{tauForGate ?? '—'}</dd>
              <dt className="text-sky/70">Features via</dt>
              <dd className="text-ice">{gate.build_info?.feature_backend ?? '—'}</dd>
              <dt className="text-sky/70">scikit-learn</dt>
              <dd className="text-ice">{gate.build_info?.sklearn_version ?? '—'}</dd>
              {gate.degenerate ? (
                <dd className="col-span-2 mt-2 rounded-xl bg-lemon px-3 py-2 text-ink">
                  Degenerate: every training label was the same class, so this gate always gives one answer.
                </dd>
              ) : null}
            </dl>
          ) : (
            <>
              <p className="mt-4 text-3xl font-bold text-ice">None yet</p>
              <p className="mt-2 text-sm text-sky">
                The benchmark's calibration step trains and saves one. Until then the GateOrchestra method is disabled.
              </p>
            </>
          )}
        </Card>

        <Card>
          <p className="font-mono text-xs uppercase tracking-widest text-navy/60">Last decision</p>
          {decision && escalateProbability != null ? (
            <div className="mt-2 flex flex-col items-center">
              <ConfidenceGauge escalateProbability={escalateProbability} threshold={learned ? threshold : null} />
              <p className="font-mono text-sm text-ink">
                <span className={decision.decision === 'STOP' ? 'text-navy' : 'text-ember'}>{decision.decision}</span> · {decision.gate_type} gate
              </p>
              <p className="mt-1 text-center text-xs text-navy/60">
                {learned
                  ? `P(escalate) = ${(escalateProbability * 100).toFixed(1)}%`
                  : decision.gate_type === 'random'
                    ? 'Random gate: a coin flip, not a probability.'
                    : 'Rule-based gate: a deterministic rule, not a probability.'}
              </p>
            </div>
          ) : (
            <p className="mt-4 text-sm text-navy/70">
              Run a gated method on{' '}
              <Link to="/run" className="font-semibold text-ember underline underline-offset-4">
                Run a Task
              </Link>{' '}
              to inspect its decision here.
            </p>
          )}
        </Card>

        <Card className="bg-lemon!">
          <p className="font-mono text-xs uppercase tracking-widest text-ink/60">Escalation threshold</p>
          <p className="mt-1 text-xs text-ink/70">Applies to GateOrchestra runs from this browser; otherwise the classifier's own decision (≈ 0.5) is used.</p>
          <label className="mt-4 flex items-center gap-2 text-sm font-medium text-ink">
            <input type="checkbox" checked={threshold != null} onChange={(e) => setThreshold(e.target.checked ? 0.5 : null)} className="accent-[#1B263B]" />
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
            className="mt-3 w-full accent-[#FF6B35] disabled:opacity-40"
            aria-label="Escalation threshold"
          />
          <p className="mt-1 font-mono text-4xl font-semibold text-ink">{threshold == null ? 'default' : threshold.toFixed(2)}</p>
          {wouldEscalate != null ? (
            <p className="mt-2 text-sm text-ink">
              At this threshold the last task would{' '}
              <span className={`rounded-full px-2 py-0.5 font-mono text-xs ${wouldEscalate ? 'bg-ember text-soft' : 'bg-navy text-cyan'}`}>
                {wouldEscalate ? 'ESCALATE' : 'STOP'}
              </span>
            </p>
          ) : null}
          <p className="mt-3 text-xs text-ink/70">τ_acc is different: it is the accuracy gap used to label training tasks, so changing it means retraining the gate.</p>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card tilt={false}>
          <h2 className="mb-3 text-xl font-bold text-ink">Feature importance</h2>
          {importance.loading ? (
            <Spinner />
          ) : importance.data?.available ? (
            <ImportanceChart rows={importance.data.importances} />
          ) : (
            <p className="rounded-2xl border border-dashed border-sky px-4 py-8 text-center text-sm text-navy/70">
              {importance.data?.reason ?? importance.error ?? 'Unavailable.'}
            </p>
          )}
        </Card>
        <Card tilt={false}>
          <h2 className="mb-3 text-xl font-bold text-ink">Features of the last gated run</h2>
          {lastRun?.features ? (
            <dl className="grid grid-cols-2 gap-2">
              {Object.entries(lastRun.features)
                .filter(([key]) => key !== 'task_id')
                .map(([key, value]) => (
                  <div key={key} className="rounded-2xl border border-sky bg-ice px-3 py-2">
                    <dt className="text-xs text-navy/60">{FEATURE_LABELS[key] ?? key}</dt>
                    <dd className="font-mono text-lg text-ink">
                      {typeof value === 'boolean' ? (value ? 'yes' : 'no') : typeof value === 'number' ? Number(value.toFixed(2)) : '—'}
                    </dd>
                  </div>
                ))}
            </dl>
          ) : (
            <EmptyState title="No gated run yet">Run RuleBasedGate, RandomGate or GateOrchestra to see the 8 features.</EmptyState>
          )}
        </Card>
      </div>
    </Page>
  )
}

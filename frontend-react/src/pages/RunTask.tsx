import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle2, ChevronDown, Dice5, Loader2, Play, Square, XCircle } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import TokenGauge from '../charts/TokenGauge'
import { Markdown } from '../components/Markdown'
import { Card, ErrorNote, Page, PageHeader } from '../components/ui'
import { api } from '../lib/api'
import { useApp } from '../lib/context'
import { FEATURE_LABELS, STRATEGY_LABELS } from '../lib/format'
import { METHODS, budgetLabel, methodInfo, runBlockReason } from '../lib/methods'
import { rise } from '../lib/motion'
import { FIELD } from '../lib/styles'
import type { MethodId, RunResult, Task } from '../lib/types'
import GateView from '../three/GateView'

const RUN_TIMEOUT_MS = 5 * 60 * 1000

// Step badge colours: pale teal = probe, teal/gold = gate side, burgundy = MAS, amber = tokens.
const BADGE = {
  sage: 'bg-sage text-ink',
  forest: 'border border-line bg-paper text-ink',
  clay: 'border border-clay/50 bg-clay/10 text-ink',
  amber: 'border border-amber/50 bg-amber/10 text-ink',
}

function Stage({ index, title, hint, badge, children }: { index: number; title: string; hint?: string; badge: keyof typeof BADGE; children: ReactNode }) {
  return (
    <motion.section
      initial={{ opacity: 0, x: -16 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.18, duration: 0.4 }}
      className="card relative p-5 pl-16 md:p-6 md:pl-16"
    >
      <span className={`absolute left-5 top-5 grid h-8 w-8 place-items-center rounded-full font-mono text-sm font-semibold md:top-6 ${BADGE[badge]}`}>
        {index + 1}
      </span>
      <h3 className="text-lg font-semibold text-ink">{title}</h3>
      {hint ? <p className="mb-3 mt-0.5 text-xs text-muted">{hint}</p> : <div className="mb-3" />}
      {children}
    </motion.section>
  )
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-forest/10">
      <motion.div className="h-full rounded-lg" style={{ background: color }} initial={{ width: 0 }} animate={{ width: `${Math.min(value, 1) * 100}%` }} transition={{ duration: 0.8 }} />
    </div>
  )
}

function FeatureGrid({ features }: { features: NonNullable<RunResult['features']> }) {
  return (
    <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
      {Object.entries(features)
        .filter(([key]) => key !== 'task_id')
        .map(([key, value]) => (
          <div key={key} className="rounded-lg border border-line bg-paper px-3 py-2">
            <dt className="text-xs text-muted">{FEATURE_LABELS[key] ?? key}</dt>
            <dd className="num text-base text-ink">
              {typeof value === 'boolean' ? (value ? 'yes' : 'no') : typeof value === 'number' ? Number(value.toFixed(2)) : '—'}
            </dd>
          </div>
        ))}
    </dl>
  )
}

function Pipeline({ result, k, probeBudget }: { result: RunResult; k: number; probeBudget: number | null }) {
  const decision = result.gate_decision
  const masCap = decision?.token_budget_cap ?? (result.method === 'Always-MAS' && probeBudget ? k * probeBudget : null)
  // Open-ended questions: show the whole response, not just the short graded part.
  const longForm = Boolean(result.full_response && result.full_response.trim().length > Math.max(200, result.predicted_answer.length + 40))
  let index = 0
  return (
    <div className="space-y-4">
      {result.probe ? (
        <Stage index={index++} title="Quick check" hint="The model answers three times; agreement between the answers is a sign it is sure (CoT-SC)." badge="sage">
          <p className="text-sm text-muted">Majority answer</p>
          <p className="text-lg font-medium text-ink">{result.probe.answer}</p>
          <div className="mt-3 flex items-center gap-3">
            <span className="w-36 shrink-0 text-sm text-muted">Self-consistency</span>
            <Bar value={result.probe.consistency_score} color="#0B4A6B" />
            <span className="font-mono text-sm text-ink">{(result.probe.consistency_score * 100).toFixed(0)}%</span>
          </div>
          <ul className="mt-3 grid gap-2 sm:grid-cols-3">
            {result.probe.samples.map((s, i) => (
              <li key={i} className="truncate rounded-md border border-line bg-paper px-3 py-1.5 font-mono text-xs" title={s}>
                <span className="text-muted">#{i + 1} </span>
                <span className={s.startsWith('Error:') ? 'text-wine' : 'text-ink'}>{s}</span>
              </li>
            ))}
          </ul>
        </Stage>
      ) : null}

      {result.features ? (
        <Stage index={index++} title="What the gate looks at" hint="Eight facts about the question and the quick check, measured before any extra work." badge="forest">
          <FeatureGrid features={result.features} />
        </Stage>
      ) : null}

      {decision ? (
        <Stage index={index++} title="Gate decision" hint="STOP keeps the quick answer. ESCALATE hands the question to the agent team." badge="forest">
          <div className="flex flex-wrap items-center gap-2">
            <span className={`chip text-xs! ${decision.decision === 'STOP' ? 'border-sage! bg-sage/25!' : 'border-clay/50! bg-clay/10!'}`}>
              {decision.decision}
            </span>
            <span className="rounded-md border border-line px-3 py-1 font-mono text-xs text-forest">{decision.gate_type} gate</span>
            <span className="rounded-md border border-line px-3 py-1 font-mono text-xs text-forest">confidence {(decision.confidence * 100).toFixed(0)}%</span>
            {decision.token_budget_cap ? (
              <span className="chip border-amber/50! bg-amber/10!">MAS cap {decision.token_budget_cap} tok</span>
            ) : null}
          </div>
          {decision.gate_type !== 'learned' ? (
            <p className="mt-2 text-xs text-muted">
              {decision.gate_type === 'random' ? 'A random gate has no real confidence; it always reports 50%.' : 'Rule-based decisions are deterministic and always report 100%.'}
            </p>
          ) : null}
        </Stage>
      ) : null}

      <Stage
        index={index++}
        title={result.mas_tokens ? `Answer from the agent team (${result.mas_strategy ? STRATEGY_LABELS[result.mas_strategy] : 'MAS'})` : 'Answer from the quick check'}
        badge={result.mas_tokens ? 'clay' : 'sage'}
      >
        {longForm ? (
          <div tabIndex={0} role="region" aria-label="Full response" className="mb-3 max-h-[32rem] overflow-auto rounded-lg border border-line bg-white/60 px-4 py-3 text-sm leading-relaxed text-ink">
            <Markdown text={result.full_response ?? ''} />
          </div>
        ) : null}
        <div className="flex flex-wrap items-center gap-3">
          {longForm ? (
            <p className="text-sm text-muted">
              Extracted answer (the part that gets graded): <span className="font-mono text-ink">{result.predicted_answer}</span>
            </p>
          ) : (
            <p className="text-xl font-semibold text-ink">{result.predicted_answer}</p>
          )}
          {result.is_correct === true ? (
            <span className="chip"><CheckCircle2 className="h-3.5 w-3.5" /> correct</span>
          ) : result.is_correct === false ? (
            <span className="chip border-wine/40! bg-wine/5! text-wine!"><XCircle className="h-3.5 w-3.5" /> incorrect</span>
          ) : (
            <span className="rounded-lg border border-line px-3 py-1 text-sm text-muted">{result.simulated ? 'simulated · ungraded' : 'ungraded (no ground truth)'}</span>
          )}
        </div>
        {result.trace && result.trace.length ? (
          <ol className="mt-4 space-y-2">
            {result.trace.map((step, i) => (
              <li key={i}>
                <details className="group rounded-lg border border-clay/40 bg-clay/5 px-4 py-2.5">
                  <summary className="flex cursor-pointer list-none items-center gap-2 text-sm">
                    <ChevronDown className="h-4 w-4 text-clay transition-transform group-open:rotate-180" />
                    <span className="font-semibold text-ink">{step.step}</span>
                    <span className="chip ml-auto">{step.tokens} tok</span>
                  </summary>
                  <pre tabIndex={0} className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap font-mono text-xs text-muted">{step.output}</pre>
                </details>
              </li>
            ))}
          </ol>
        ) : result.mas_tokens && result.simulated ? (
          <p className="mt-2 text-xs text-muted">The offline simulator has no sub-agent steps to show.</p>
        ) : null}
      </Stage>

      <Stage index={index} title="What it cost" hint="Tokens are the unit models charge in. The agent team is capped at a multiple of the quick check." badge="amber">
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:items-start">
          <TokenGauge probeTokens={result.probe_tokens} masTokens={result.mas_tokens} cap={masCap} size={180} />
          <dl className="grid flex-1 grid-cols-2 gap-x-3 gap-y-2 font-mono text-sm">
            <dt className="text-muted">Total spent</dt>
            <dd className="text-ink">{result.tokens_spent}</dd>
            <dt className="text-muted">Path</dt>
            <dd className="text-ink">{result.mas_tokens ? 'escalated to the agent team' : 'answered by the quick check'}</dd>
            <dt className="text-muted">Latency</dt>
            <dd className="text-ink">{result.latency_ms != null ? `${(result.latency_ms / 1000).toFixed(1)} s` : '—'}</dd>
            <dt className="text-muted">Quick-check budget</dt>
            <dd className="text-ink">{probeBudget ? `${probeBudget} completion tok` : '—'}</dd>
          </dl>
        </div>
      </Stage>
    </div>
  )
}

export default function RunTask() {
  const { models, lastRun, setLastRun, threshold, backend } = useApp()
  const gateAvailable = Boolean(models?.trained_gate)
  const [question, setQuestion] = useState('')
  const [context, setContext] = useState('')
  const [groundTruth, setGroundTruth] = useState('')
  const [chosenMethod, setMethod] = useState<MethodId | null>(null)
  // Default to the learned gate once one exists; never leave a disabled option selected.
  const method: MethodId =
    chosenMethod && !(chosenMethod === 'GateOrchestra' && !gateAvailable)
      ? chosenMethod
      : gateAvailable
        ? 'GateOrchestra'
        : 'CoT-SC'
  const [k, setK] = useState(3)
  const [simulate, setSimulate] = useState(false)
  const [running, setRunning] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<RunResult | null>(lastRun)
  const [runKey, setRunKey] = useState(0)
  const [taskPool, setTaskPool] = useState<Task[]>([])
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!running) return
    const started = Date.now()
    const id = setInterval(() => setElapsed(Math.round((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(id)
  }, [running])

  const pickTask = async () => {
    try {
      const pool = taskPool.length ? taskPool : (await api.tasks('val')).tasks
      if (!taskPool.length) setTaskPool(pool)
      const task = pool[Math.floor(Math.random() * pool.length)]
      setQuestion(task.question)
      setContext(task.context ?? '')
      setGroundTruth(task.ground_truth ?? '')
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const run = async () => {
    if (!question.trim() || running) return
    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort('timeout'), RUN_TIMEOUT_MS)
    setRunning(true)
    setElapsed(0)
    setError(null)
    setResult(null)
    setRunKey((n) => n + 1)
    try {
      const data = await api.run(
        {
          task_id: `ui-${Date.now()}`,
          question: question.trim(),
          context: context.trim() || null,
          ground_truth: groundTruth.trim() || null,
          method,
          k,
          force_simulation: simulate,
          threshold: method === 'GateOrchestra' ? threshold : null,
        },
        controller.signal,
      )
      setResult(data)
      setLastRun(data)
    } catch (err) {
      if (controller.signal.aborted) {
        setError(
          `${controller.signal.reason === 'timeout' ? 'Run timed out after 5 minutes' : 'Run cancelled'}. The server may still finish it; it will then appear in the Token Accountant.`,
        )
      } else {
        setError((err as Error).message)
      }
    } finally {
      clearTimeout(timeout)
      abortRef.current = null
      setRunning(false)
    }
  }

  const decision = result ? (result.gate_decision?.decision ?? (result.mas_tokens ? 'ESCALATE' : 'STOP')) : null
  const blockReason = runBlockReason(question, backend, running)

  return (
    <Page>
      <PageHeader kicker="Live pipeline" title="Run a task">
        Send one question through the real pipeline and watch each step. Offline simulation tests
        the routing without calling a model and is never graded.
      </PageHeader>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        <div className="space-y-4">
          <Card tilt={false}>
            <label className="text-sm font-medium text-forest" htmlFor="question">Question</label>
            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) run()
              }}
              rows={3}
              placeholder="Ask a multi-step reasoning question…"
              className={`${FIELD} mt-1.5 resize-y text-base`}
            />
            <details className="mt-3">
              <summary className="cursor-pointer text-sm text-muted">Context and ground truth (optional)</summary>
              <textarea value={context} onChange={(e) => setContext(e.target.value)} rows={2} placeholder="Supporting passage" className={`${FIELD} mt-2 text-sm`} />
              <input value={groundTruth} onChange={(e) => setGroundTruth(e.target.value)} placeholder="Ground truth (enables grading)" className={`${FIELD} mt-2 text-sm`} />
            </details>

            <div className="mt-4 space-y-3">
              <label className="block text-sm font-medium text-forest">
                How to answer
                <select value={method} onChange={(e) => setMethod(e.target.value as MethodId)} className={`${FIELD} mt-1.5`} aria-describedby="method-summary">
                  {METHODS.map((m) => (
                    <option key={m.id} value={m.id} disabled={m.id === 'GateOrchestra' && !gateAvailable}>
                      {m.label}
                      {m.id === 'GateOrchestra' && !gateAvailable ? ' (no trained gate yet)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <p id="method-summary" className="-mt-1 text-xs text-muted">{methodInfo(method).summary}</p>
              <label className="block text-sm font-medium text-forest">
                Budget for the agent team
                <select value={k} onChange={(e) => setK(Number(e.target.value))} className={`${FIELD} mt-1.5`}>
                  {(models?.k_values ?? [2, 3, 5]).map((v) => (
                    <option key={v} value={v}>{budgetLabel(v)}</option>
                  ))}
                </select>
              </label>
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2.5">
              <label className={`inline-flex cursor-pointer items-center gap-2 rounded-lg border px-3.5 py-2 text-sm ${simulate ? 'border-forest bg-forest text-paper' : 'border-line text-forest'}`}>
                <input type="checkbox" checked={simulate} onChange={(e) => setSimulate(e.target.checked)} className="accent-[#0B4A6B]" />
                Offline simulation
              </label>
              {method === 'GateOrchestra' && threshold != null ? (
                <span className="chip">threshold {threshold.toFixed(2)}</span>
              ) : null}
              <button onClick={pickTask} title="Fill in a random question from the validation set, with its answer so the run can be graded" className="inline-flex items-center gap-1.5 rounded-lg border border-line px-3.5 py-2 text-sm text-forest hover:bg-forest/5">
                <Dice5 className="h-4 w-4" /> Try a benchmark question
              </button>
              <div className="ml-auto flex gap-2">
                {running ? (
                  <button onClick={() => abortRef.current?.abort('cancelled')} className="inline-flex items-center gap-1.5 rounded-lg border border-wine px-4 py-2 text-sm text-wine hover:bg-wine/10">
                    <Square className="h-4 w-4" /> Cancel
                  </button>
                ) : null}
                <button
                  onClick={run}
                  disabled={!!blockReason || running}
                  aria-describedby="run-hint"
                  className="btn btn-accent px-5!"
                >
                  {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {running ? `Running… ${elapsed}s` : 'Run'}
                </button>
              </div>
            </div>
            {blockReason ? <p id="run-hint" className="mt-2 text-right text-xs text-muted">{blockReason}</p> : null}
          </Card>

          {error ? <ErrorNote message={error} /> : null}

          <AnimatePresence mode="wait">
            {result ? (
              <motion.div key={result.task_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                <Pipeline result={result} k={k} probeBudget={models?.probe_token_budget ?? null} />
              </motion.div>
            ) : null}
          </AnimatePresence>
        </div>

        <motion.div variants={rise} className="panel sticky top-20 h-fit overflow-hidden">
          <GateView
            running={running}
            decision={running ? null : decision}
            strategy={running ? null : (result?.mas_strategy ?? null)}
            runKey={runKey}
            demo={!running && !result}
            className="h-[360px] md:h-[480px]"
          />
          <p className="border-t border-paper/10 px-5 py-3 font-mono text-xs text-sage">
            {running ? (
              <span className="text-butter">Probing… the orb waits at the gate until the decision arrives.</span>
            ) : result ? (
              <>
                <span className={decision === 'STOP' ? 'text-saffron' : 'text-clay-light'}>{decision}</span>
                {result.mas_strategy ? ` → ${STRATEGY_LABELS[result.mas_strategy]}` : ''} · {result.tokens_spent} tokens
              </>
            ) : (
              'Illustrative loop until you run a task.'
            )}
          </p>
        </motion.div>
      </div>
    </Page>
  )
}

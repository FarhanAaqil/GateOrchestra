import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { Card, CountUp, NoCitableResults, Page, PillLink, Spinner, StatusBadge, Tag } from '../components/ui'
import { api } from '../lib/api'
import { useApp } from '../lib/context'
import { isCitable } from '../lib/format'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'
import GateView from '../three/GateView'

const STEPS = [
  { title: 'Probe', text: 'A cheap CoT-SC pass: three samples, a majority answer, and how much they agree.' },
  { title: 'Features', text: 'Eight pre-execution signals: agreement, probe tokens, entities, clauses, depth and parallelism proxies.' },
  { title: 'Gate', text: 'STOP returns the probe answer. ESCALATE hands the task to the multi-agent orchestrator.' },
  { title: 'Capped MAS', text: 'One sub-agent (ReAct, Debate or Reflexion) answers within k × the probe’s tokens.' },
]

const loadSplitSizes = async () => {
  const [train, val, test] = await Promise.all([api.tasks('train', 1), api.tasks('val', 1), api.tasks('test', 1)])
  return { train: train.total, val: val.total, test: test.total }
}

/** Scrolling strip of real system facts (from the API, not hardcoded). */
function FactTicker() {
  const { models, reducedMotion } = useApp()
  const sizes = useApi(loadSplitSizes)
  const facts = [
    sizes.data ? `${sizes.data.train + sizes.data.val + sizes.data.test} benchmark tasks` : null,
    sizes.data ? `${sizes.data.train} / ${sizes.data.val} / ${sizes.data.test} train · val · test` : null,
    '8 gate features',
    '3 sub-agents: ReAct · Debate · Reflexion',
    models ? `k ∈ {${models.k_values.join(', ')}}` : null,
    models ? `probe budget ${models.probe_token_budget} completion tokens` : null,
    models ? `${models.default_model} via ${models.current_provider}` : null,
  ].filter(Boolean) as string[]
  const row = [...facts, ...facts]
  return (
    <motion.div variants={rise} className="relative mt-6 overflow-hidden rounded-full bg-navy py-3">
      <div
        className="flex w-max gap-10 whitespace-nowrap px-6 font-mono text-sm text-ice"
        style={{ animation: reducedMotion ? undefined : 'marquee 38s linear infinite' }}
      >
        {row.map((fact, i) => (
          <span key={i} className="inline-flex items-center gap-3">
            <span className={`h-2 w-2 rounded-full ${i % 3 === 0 ? 'bg-cyan' : i % 3 === 1 ? 'bg-ember' : 'bg-lemon'}`} />
            {fact}
          </span>
        ))}
      </div>
    </motion.div>
  )
}

export default function Overview() {
  const summary = useApi(api.summary)
  const status = summary.data?.status
  const rows = summary.data?.aggregated ?? []
  const gate = rows.find((r) => r.method === 'GateOrchestra')
  const mas = rows.find((r) => r.method === 'Always-MAS')
  const tasksEvaluated = gate?.n_tasks_per_seed?.reduce((a, b) => a + b, 0)

  return (
    <Page>
      <section className="grid gap-5 lg:grid-cols-12">
        <motion.div variants={rise} className="card flex flex-col justify-between p-7 md:p-10 lg:col-span-7">
          <div>
            <p className="font-mono text-sm text-navy/60">
              <span className="text-ember">01</span> — Token-budget-calibrated gating
            </p>
            <h1 className="mt-4 text-5xl font-bold leading-[0.98] text-ink md:text-7xl">
              Decide <span className="mark-lemon">whether</span> to orchestrate.
            </h1>
            <p className="mt-5 max-w-xl text-lg text-navy/75">
              A cheap CoT-SC probe answers first. A small trained gate escalates to the multi-agent orchestrator only when
              decomposition is likely to beat that answer at an equal token budget.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <Tag>Pre-execution</Tag>
              <Tag>Equal budgets</Tag>
              <Tag>Honest accounting</Tag>
            </div>
          </div>
          <div className="mt-8 flex flex-wrap gap-3">
            <PillLink to="/run" variant="navy">
              Run a task <ArrowRight className="h-4 w-4" />
            </PillLink>
            <PillLink to="/results" variant="ghost">
              See results
            </PillLink>
          </div>
        </motion.div>

        <motion.div variants={rise} className="panel relative overflow-hidden lg:col-span-5">
          <GateView running={false} decision={null} strategy={null} runKey={0} demo className="h-[360px] lg:h-full lg:min-h-[440px]" />
          <span className="absolute left-5 top-4 rounded-full bg-ice/10 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-sky">
            Illustrative loop · live on Run a Task
          </span>
        </motion.div>
      </section>

      <FactTicker />

      <section className="mt-12">
        <motion.div variants={rise} className="mb-4 flex items-center gap-3">
          <h2 className="text-3xl font-bold text-ink">
            <span className="font-mono text-base text-ember">02 </span>Key results
          </h2>
          {status ? <StatusBadge status={status} /> : null}
        </motion.div>
        {summary.loading ? (
          <Spinner />
        ) : isCitable(status) && gate && mas ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card navy>
              <p className="text-sm text-sky">Task runs evaluated</p>
              <CountUp value={tasksEvaluated ?? 0} className="mt-3 block text-5xl font-semibold text-ice" />
              <p className="mt-2 font-mono text-xs text-sky/70">{gate.n_seeds} seed(s)</p>
            </Card>
            <Card>
              <p className="text-sm text-navy/60">Tokens saved vs Always-MAS</p>
              <CountUp value={gate.token_savings_pct_mean} decimals={1} suffix="%" className="mt-3 block text-5xl font-semibold text-ember" />
            </Card>
            <Card>
              <p className="text-sm text-navy/60">Accuracy vs Always-MAS</p>
              <CountUp value={gate.accuracy_mean - mas.accuracy_mean} decimals={1} suffix=" pts" className="mt-3 block text-5xl font-semibold text-ink" />
              <p className="mt-2 font-mono text-xs text-navy/60">
                {gate.accuracy_mean.toFixed(1)}% vs {mas.accuracy_mean.toFixed(1)}%
              </p>
            </Card>
            <Card className="bg-lemon!">
              <p className="text-sm text-ink/70">Escalation rate</p>
              <CountUp value={(gate.escalate_rate_mean ?? 0) * 100} decimals={1} suffix="%" className="mt-3 block text-5xl font-semibold text-ink" />
            </Card>
          </div>
        ) : (
          <NoCitableResults status={status} what="results" />
        )}
      </section>

      <section className="mt-12">
        <motion.h2 variants={rise} className="mb-4 text-3xl font-bold text-ink">
          <span className="font-mono text-base text-ember">03 </span>How a task flows
        </motion.h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {STEPS.map(({ title, text }, i) => (
            <Card key={title} navy={i === 2}>
              <p className={`font-mono text-5xl font-semibold ${i === 2 ? 'text-cyan' : i === 3 ? 'text-ember' : 'text-sky'}`}>0{i + 1}</p>
              <h3 className={`mt-4 text-xl font-bold ${i === 2 ? 'text-ice' : 'text-ink'}`}>{title}</h3>
              <p className={`mt-2 text-sm ${i === 2 ? 'text-sky' : 'text-navy/70'}`}>{text}</p>
            </Card>
          ))}
        </div>
      </section>
    </Page>
  )
}

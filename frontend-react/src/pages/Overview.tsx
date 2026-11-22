import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { CountUp, NoCitableResults, Page, PillLink, SectionTitle, SkeletonCards, Stat, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import { useApp } from '../lib/context'
import { isCitable } from '../lib/format'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

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

/** The pipeline as a diagram: question, probe, gate, then one of two paths. */
function PipelineDiagram() {
  const node = 'rounded-md border border-paper/15 bg-paper/[0.04] px-3 py-2'
  return (
    <div className="panel flex h-full flex-col justify-between p-6" aria-label="How GateOrchestra routes a question">
      <p className="eyebrow text-sage/70!">Pipeline</p>
      <div className="mt-6 space-y-3 font-mono text-[12px]">
        <div className={node}>
          <p className="text-sage/70">input</p>
          <p className="text-paper">question</p>
        </div>
        <div className="ml-4 h-4 w-px bg-paper/20" aria-hidden />
        <div className={node}>
          <p className="text-sage/70">probe · CoT-SC × 3</p>
          <p className="text-paper">answer + agreement</p>
        </div>
        <div className="ml-4 h-4 w-px bg-paper/20" aria-hidden />
        <div className="rounded-md border border-saffron/50 bg-saffron/[0.06] px-3 py-2">
          <p className="text-saffron">gate · 8 features</p>
          <p className="text-paper">is decomposition worth it?</p>
        </div>
        <div className="grid grid-cols-2 gap-3 pt-3">
          <div className={node}>
            <p className="text-sage">STOP</p>
            <p className="text-paper/80">return the probe answer</p>
          </div>
          <div className="rounded-md border border-clay/50 bg-clay/[0.07] px-3 py-2">
            <p className="text-clay-light">ESCALATE</p>
            <p className="text-paper/80">MAS within k × probe tokens</p>
          </div>
        </div>
      </div>
    </div>
  )
}

/** The system's configuration, read from the API rather than written into the page. */
function SpecSheet() {
  const { models } = useApp()
  const sizes = useApi(loadSplitSizes)
  const rows: [string, string | null][] = [
    ['Benchmark tasks', sizes.data ? String(sizes.data.train + sizes.data.val + sizes.data.test) : null],
    ['Splits', sizes.data ? `${sizes.data.train} · ${sizes.data.val} · ${sizes.data.test}` : null],
    ['Gate features', '8'],
    ['Sub-agents', 'ReAct · Debate · Reflexion'],
    ['k values', models ? models.k_values.join(', ') : null],
    ['Probe budget', models ? `${models.probe_token_budget} tokens` : null],
    ['Benchmark model', models ? `${models.default_model} (${models.current_provider})` : null],
  ]
  return (
    <motion.dl variants={rise} className="card mt-6 grid grid-cols-2 divide-forest/10 overflow-hidden sm:grid-cols-4 lg:grid-cols-7 lg:divide-x">
      {rows.map(([label, value]) => (
        <div key={label} className="border-b border-forest/10 px-4 py-3 lg:border-b-0">
          <dt className="eyebrow">{label}</dt>
          <dd className="num mt-1 truncate text-sm text-ink" title={value ?? undefined}>{value ?? '—'}</dd>
        </div>
      ))}
    </motion.dl>
  )
}

export default function Overview() {
  const { isAdmin } = useApp()
  const summary = useApi(api.summary)
  const status = summary.data?.status
  const rows = summary.data?.aggregated ?? []
  const gate = rows.find((r) => r.method === 'GateOrchestra')
  const mas = rows.find((r) => r.method === 'Always-MAS')
  const tasksEvaluated = gate?.n_tasks_per_seed?.reduce((a, b) => a + b, 0)

  return (
    <Page>
      <section className="grid gap-6 lg:grid-cols-12">
        <motion.div variants={rise} className="flex flex-col justify-center py-4 lg:col-span-7">
          <p className="eyebrow">Research · token-budget-calibrated gating</p>
          <h1 className="mt-3 text-4xl font-semibold leading-[1.08] tracking-tight text-ink md:text-[52px]">
            Decide whether to orchestrate, before paying for it.
          </h1>
          <p className="mt-5 max-w-xl text-[17px] leading-relaxed text-forest/70">
            A cheap CoT-SC probe answers first. A small trained gate escalates to the multi-agent orchestrator only when
            decomposition is likely to beat that answer at an equal token budget.
          </p>
          <div className="mt-7 flex flex-wrap gap-2.5">
            {isAdmin ? (
              <PillLink to="/run" variant="forest">
                Run a task <ArrowRight className="h-4 w-4" />
              </PillLink>
            ) : null}
            <PillLink to="/results" variant={isAdmin ? 'ghost' : 'forest'}>
              See the results {isAdmin ? null : <ArrowRight className="h-4 w-4" />}
            </PillLink>
          </div>
        </motion.div>
        <motion.div variants={rise} className="lg:col-span-5">
          <PipelineDiagram />
        </motion.div>
      </section>

      <SpecSheet />

      <SectionTitle aside={status ? <StatusBadge status={status} /> : null}>Key results</SectionTitle>
      {summary.loading ? (
        <SkeletonCards count={4} height="h-10" />
      ) : isCitable(status) && gate && mas ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat label="Task runs evaluated" value={<CountUp value={tasksEvaluated ?? 0} />} note={`${gate.n_seeds} seed(s)`} />
          <Stat label="Tokens vs Always-MAS" tone="clay" value={<CountUp value={gate.token_savings_pct_mean} decimals={1} suffix="%" />} note="saved per task, on average" />
          <Stat
            label="Accuracy vs Always-MAS"
            value={<CountUp value={gate.accuracy_mean - mas.accuracy_mean} decimals={1} suffix=" pts" />}
            note={`${gate.accuracy_mean.toFixed(1)}% vs ${mas.accuracy_mean.toFixed(1)}%`}
          />
          <Stat label="Escalation rate" value={<CountUp value={(gate.escalate_rate_mean ?? 0) * 100} decimals={1} suffix="%" />} note="tasks sent to the multi-agent system" />
        </div>
      ) : (
        <NoCitableResults status={status} what="results" />
      )}

      <SectionTitle>How a task flows</SectionTitle>
      <motion.ol variants={rise} className="card grid overflow-hidden sm:grid-cols-2 lg:grid-cols-4">
        {STEPS.map(({ title, text }, i) => (
          <li key={title} className="border-forest/10 p-5 max-lg:border-b sm:max-lg:odd:border-r lg:border-r lg:last:border-r-0">
            <p className={`num text-xs ${i === 2 ? 'font-semibold text-ink' : 'text-forest/70'}`}>0{i + 1}</p>
            <h3 className="mt-2 text-[15px] font-semibold text-ink">{title}</h3>
            <p className="mt-1.5 text-sm leading-relaxed text-forest/70">{text}</p>
          </li>
        ))}
      </motion.ol>
    </Page>
  )
}

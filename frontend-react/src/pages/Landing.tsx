import { motion } from 'framer-motion'
import { ArrowRight, Coins, Gauge, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { PlanCards } from '../components/PlanCards'
import { Card, ErrorNote, Page, SkeletonCards } from '../components/ui'
import { usageApi } from '../lib/account'
import { useApp } from '../lib/context'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

const REPORT_URL = 'https://github.com/FarhanAaqil/GateOrchestra/blob/main/reports/research_report.md'

// Numbers from the README and reports/final_analysis.md (verified benchmark runs). Keep in step.
const RESULTS = [
  { value: '-33%', label: 'tokens vs always running the multi-agent system', note: '784 vs 1,171 tokens per task, Qwen2.5-3B, 96 test tasks x 2 seeds' },
  { value: '97.9%', label: 'accuracy with the gate', note: 'vs 92.2% always running the multi-agent system on the same tasks' },
  { value: '-54%', label: 'tokens on gpt-oss-120b', note: 'reduced run (1 seed): 99.0% accuracy, the same as the single-agent baseline' },
]

const STEPS = [
  { icon: Gauge, title: 'A quick check first', text: 'A cheap model answers a few times. If the answers agree, that is usually enough.' },
  { icon: ShieldCheck, title: 'A trained gate decides', text: 'The gate looks at that agreement and the question, and escalates only when it expects research to help.' },
  { icon: Coins, title: 'You pay for effort you need', text: 'Hard questions get a research agent with web lookups, Python and your documents; easy ones do not.' },
]

export default function Landing() {
  const { user } = useApp()
  const plans = useApi(usageApi.plans)
  const start = user ? '/chat' : '/signup'
  return (
    <Page>
      <motion.section variants={rise} className="mx-auto max-w-4xl py-8 text-center md:py-14">
        <p className="font-mono text-xs uppercase tracking-widest text-wine">GateOrchestra</p>
        <h1 className="mt-3 text-4xl font-bold leading-tight text-ink md:text-6xl">
          An assistant that decides <span className="mark-lemon">how hard to think</span>
        </h1>
        <p className="mx-auto mt-4 max-w-2xl text-lg text-navy/75">
          Most questions do not need an expensive multi-agent run. GateOrchestra answers the easy ones directly and sends only the
          hard ones to a research agent, so you get careful answers without paying for effort that adds nothing.
        </p>
        <div className="mt-7 flex flex-wrap justify-center gap-3">
          <Link to={start} className="inline-flex items-center gap-2 rounded-full bg-ember px-6 py-3 text-sm font-semibold text-soft glow-ember">
            {user ? 'Open the chat' : 'Sign up free'} <ArrowRight className="h-4 w-4" />
          </Link>
          <Link to="/results" className="inline-flex items-center gap-2 rounded-full border border-navy/20 px-6 py-3 text-sm font-semibold text-navy hover:bg-soft">
            See the benchmark
          </Link>
        </div>
      </motion.section>

      <div className="grid gap-4 md:grid-cols-3">
        {STEPS.map(({ icon: Icon, title, text }) => (
          <Card key={title}>
            <Icon className="h-6 w-6 text-wine" aria-hidden />
            <h2 className="mt-3 text-lg font-bold text-ink">{title}</h2>
            <p className="mt-1 text-sm text-navy/70">{text}</p>
          </Card>
        ))}
      </div>

      <motion.h2 variants={rise} className="mb-4 mt-12 text-3xl font-bold text-ink">
        <span className="font-mono text-base text-wine">Measured </span>What the research found
      </motion.h2>
      <div className="grid gap-4 md:grid-cols-3">
        {RESULTS.map((r) => (
          <Card key={r.value} navy>
            <p className="font-mono text-4xl font-semibold text-lemon">{r.value}</p>
            <p className="mt-2 font-semibold text-ice">{r.label}</p>
            <p className="mt-1 text-xs text-sky">{r.note}</p>
          </Card>
        ))}
      </div>
      <motion.p variants={rise} className="mt-4 max-w-4xl text-sm text-navy/70">
        These come from the project's benchmark (96 held-out test tasks), not from chats on this site. On the 3B model the multi-agent system was simply not better
        than the quick check, so the gate learned to escalate only 3% of tasks; the goal of 40% savings was not reached. Your own
        questions will differ, which is why the <Link to="/usage" className="underline">usage page</Link> shows how your answers were
        actually routed. Read the <a href={REPORT_URL} className="underline" target="_blank" rel="noreferrer">full research report</a>{' '}
        or explore the <Link to="/results" className="underline">results</Link>.
      </motion.p>

      <motion.h2 variants={rise} className="mb-4 mt-12 text-3xl font-bold text-ink" id="pricing">
        <span className="font-mono text-base text-wine">Pricing </span>Start free
      </motion.h2>
      {plans.error ? <ErrorNote message={plans.error} /> : null}
      {plans.loading && !plans.data ? <SkeletonCards count={2} height="h-48" /> : null}
      {plans.data ? (
        <PlanCards
          plans={plans.data.plans}
          action={(plan) => (
            <Link
              to={user ? (plan.price_usd_month > 0 ? '/pricing' : '/chat') : '/signup'}
              className={`inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold ${plan.price_usd_month > 0 ? 'bg-lemon text-ink' : 'bg-ink text-soft'}`}
            >
              {user ? (plan.price_usd_month > 0 ? 'Upgrade' : 'Open the chat') : 'Sign up'} <ArrowRight className="h-4 w-4" />
            </Link>
          )}
        />
      ) : null}
    </Page>
  )
}

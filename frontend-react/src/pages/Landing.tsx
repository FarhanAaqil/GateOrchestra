import { motion } from 'framer-motion'
import { ArrowRight, ArrowUpRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { PlanCards } from '../components/PlanCards'
import { ErrorNote, Page, SkeletonCards } from '../components/ui'
import { usageApi } from '../lib/account'
import { useApp } from '../lib/context'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

const REPORT_URL = 'https://github.com/FarhanAaqil/GateOrchestra/blob/main/reports/research_report.md'

// Numbers from the README and reports/final_analysis.md (verified benchmark runs). Keep in step.
const RESULTS = [
  { value: '−33%', label: 'tokens vs always running the multi-agent system', note: '784 vs 1,171 tokens per task · Qwen2.5-3B · 96 test tasks × 2 seeds' },
  { value: '97.9%', label: 'accuracy with the gate', note: 'vs 92.2% always running the multi-agent system on the same tasks' },
  { value: '−54%', label: 'tokens on gpt-oss-120b', note: 'reduced run (1 seed) · 99.0% accuracy, equal to the single-agent baseline' },
]

// Each step wears the colour it has in the logo and the charts: sage quick check,
// saffron gate, clay escalation.
const STEP_TONES = [
  { bar: 'bg-sage', badge: 'bg-sage text-forest' },
  { bar: 'bg-saffron', badge: 'bg-saffron text-ink' },
  { bar: 'bg-clay', badge: 'bg-clay-deep text-paper' },
]

const STEPS = [
  { title: 'A quick check first', text: 'A cheap model answers a few times. When the answers agree, that is usually enough.' },
  { title: 'A trained gate decides', text: 'It reads that agreement and the shape of the question, and escalates only when research is likely to help.' },
  { title: 'Effort only where it pays', text: 'Hard questions get a research agent with web lookups, Python and your documents. Easy ones do not.' },
]

/** A sketch of one answer in the product: what people see under each reply. */
function AnswerPreview() {
  return (
    <div className="card overflow-hidden" aria-hidden>
      <div className="flex items-center gap-1.5 border-b border-forest/10 px-4 py-2.5">
        <span className="h-2 w-2 rounded-full bg-forest/15" />
        <span className="h-2 w-2 rounded-full bg-forest/15" />
        <span className="h-2 w-2 rounded-full bg-forest/15" />
        <span className="ml-2 text-xs text-forest/70">Example conversation</span>
      </div>
      <div className="space-y-4 p-5 text-sm">
        <p className="ml-auto w-fit max-w-[80%] rounded-lg bg-forest px-3.5 py-2 text-paper">What is the capital of Australia?</p>
        <div>
          <p className="text-ink">Canberra. It was chosen in 1908 as a compromise between Sydney and Melbourne.</p>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <span className="chip border-sage! bg-sage/25!">Answered directly</span>
            <span className="chip">212 tokens</span>
            <span className="chip">0.9 s</span>
          </div>
        </div>
        <p className="ml-auto w-fit max-w-[80%] rounded-lg bg-forest px-3.5 py-2 text-paper">Write a function that checks whether a number is prime, and test it.</p>
        <div>
          <p className="text-ink">Here is a version that only tries divisors up to the square root, with tests that passed in the sandbox …</p>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            <span className="chip border-clay/50! bg-clay/10!">Researched with tools</span>
            <span className="chip">ran Python</span>
            <span className="chip">2,940 tokens</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function Landing() {
  const { user } = useApp()
  const plans = useApi(usageApi.plans)
  const start = user ? '/chat' : '/signup'
  return (
    <Page>
      <section className="grid items-center gap-10 py-6 lg:grid-cols-12 lg:py-12">
        <motion.div variants={rise} className="lg:col-span-6">
          <p className="eyebrow">AI assistant · research-backed routing</p>
          <h1 className="mt-4 text-4xl font-semibold leading-[1.06] tracking-tight text-ink md:text-[56px]">
            An assistant that knows how hard to <em className="font-medium italic text-clay-deep">think.</em>
          </h1>
          <p className="mt-5 max-w-lg text-[17px] leading-relaxed text-forest/70">
            Most questions do not need an expensive multi-agent run. GateOrchestra answers the easy ones directly and sends only the
            hard ones to a research agent, so you get careful answers without paying for effort that adds nothing.
          </p>
          <div className="mt-8 flex flex-wrap gap-2.5">
            <Link to={start} className="btn btn-accent px-5! py-2.5!">
              {user ? 'Open the chat' : 'Start free'} <ArrowRight className="h-4 w-4" />
            </Link>
            <Link to="/results" className="btn btn-secondary px-5! py-2.5!">See the benchmark</Link>
          </div>
          <p className="mt-4 text-xs text-forest/70">Free plan · no card needed</p>
        </motion.div>
        <motion.div variants={rise} className="lg:col-span-6">
          <AnswerPreview />
        </motion.div>
      </section>

      <motion.ol variants={rise} className="mt-6 grid gap-px overflow-hidden rounded-xl border border-forest/10 bg-forest/10 md:grid-cols-3">
        {STEPS.map(({ title, text }, i) => (
          <li key={title} className="relative bg-paper p-6">
            <span className={`absolute inset-x-0 top-0 h-1 ${STEP_TONES[i].bar}`} aria-hidden />
            <p className={`num inline-grid h-6 w-6 place-items-center rounded-full text-xs font-semibold ${STEP_TONES[i].badge}`}>{i + 1}</p>
            <h2 className="mt-2 text-[15px] font-semibold text-ink">{title}</h2>
            <p className="mt-1.5 text-sm leading-relaxed text-forest/70">{text}</p>
          </li>
        ))}
      </motion.ol>

      <section className="mt-16">
        <motion.div variants={rise} className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <p className="eyebrow">Measured</p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink">What the research found</h2>
          </div>
          <a href={REPORT_URL} target="_blank" rel="noreferrer" className="btn btn-ghost">
            Read the report <ArrowUpRight className="h-4 w-4" />
          </a>
        </motion.div>
        <motion.div variants={rise} className="panel mt-5 grid overflow-hidden md:grid-cols-3">
          {RESULTS.map((r) => (
            <div key={r.value} className="border-paper/10 p-6 max-md:border-b md:border-r md:last:border-r-0">
              <p className="num text-[40px] font-medium leading-none text-saffron">{r.value}</p>
              <p className="mt-3 text-sm font-medium text-paper">{r.label}</p>
              <p className="mt-1 text-xs leading-relaxed text-sage/70">{r.note}</p>
            </div>
          ))}
        </motion.div>
        <motion.p variants={rise} className="mt-4 max-w-3xl text-sm leading-relaxed text-forest/70">
          These come from the project's benchmark (96 held-out test tasks), not from chats on this site. On the 3B model the
          multi-agent system was simply not better than the quick check, so the gate learned to escalate only 3% of tasks; the goal
          of 40% savings was not reached, and the gate is not measurably better than never escalating. Its job there was to avoid waste. Your own questions will differ, which is why the{' '}
          <Link to="/usage" className="text-ink underline decoration-forest/30 underline-offset-2">usage page</Link> shows how your
          answers were actually routed. Explore the <Link to="/results" className="text-ink underline decoration-forest/30 underline-offset-2">full results</Link>.
        </motion.p>
      </section>

      <section className="mt-16" id="pricing">
        <motion.div variants={rise}>
          <p className="eyebrow">Pricing</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-ink">Start free, upgrade when it earns its place</h2>
        </motion.div>
        <div className="mt-5">
          {plans.error ? <ErrorNote message={plans.error} /> : null}
          {plans.loading && !plans.data ? <SkeletonCards count={2} height="h-48" /> : null}
          {plans.data ? (
            <PlanCards
              plans={plans.data.plans}
              action={(plan) => (
                <Link
                  to={user ? (plan.price_usd_month > 0 ? '/pricing' : '/chat') : '/signup'}
                  className={`btn w-full ${plan.price_usd_month > 0 ? 'btn-accent' : 'btn-secondary'}`}
                >
                  {user ? (plan.price_usd_month > 0 ? 'Upgrade' : 'Open the chat') : plan.price_usd_month > 0 ? 'Start with Pro' : 'Start free'}
                </Link>
              )}
            />
          ) : null}
        </div>
      </section>
    </Page>
  )
}

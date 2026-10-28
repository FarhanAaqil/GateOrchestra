import { motion } from 'framer-motion'
import { Card, Page, PageHeader, Tag } from '../components/ui'
import { rise } from '../lib/motion'

const QUESTIONS = [
  { id: 'RQ1', text: 'Can a lightweight, trainable gate predict, from task-level features available before execution, whether multi-agent decomposition will beat a token-matched CoT-SC baseline on a given task?' },
  { id: 'RQ2', text: 'Does inserting this gate in front of an orchestrator reduce the token-matched accuracy gap IlluMAS documents, without materially hurting accuracy on tasks that genuinely benefit from decomposition?' },
  { id: 'RQ3', text: 'How does gate accuracy vary across the MASBench structural axes? Does its confidence correlate with Depth and Parallel scores as IlluMAS suggests?' },
]

const TEAM = [
  { name: 'Farhan Aaqil Durrani', role: 'Team Lead' },
  { name: 'Kounain Sanaliya Khan', role: 'Team member' },
  { name: 'Shaik Mohammed Umaruddin', role: 'Team member' },
  { name: 'Muzra Fatima', role: 'Team member' },
]

const initials = (name: string) =>
  name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join('')

export default function About() {
  return (
    <Page>
      <PageHeader number="08" kicker="About" title={<>Why a gate <span className="mark-lemon">in front</span> of the orchestra</>}>
        A capstone research project on when multi-agent decomposition is worth its tokens.
      </PageHeader>

      <Card navy className="md:p-10">
        <p className="font-mono text-xs uppercase tracking-widest text-cyan">The research gap</p>
        <p className="mt-4 max-w-4xl text-xl leading-relaxed text-ice md:text-2xl">
          Multi-agent systems such as MAS-Orchestra often lose to a well-tuned single-agent CoT-SC baseline once thinking-token
          budgets are held equal, at up to <span className="text-ember">10× the cost</span>. No published orchestrator decides,
          before spending multi-agent compute, whether decomposition is worth it at all. <span className="text-lemon">GateOrchestra is that decision.</span>
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="navy">IlluMAS, 2026</Tag>
          <Tag tone="navy">Tran &amp; Kiela, 2026</Tag>
          <Tag tone="navy">MAS-Orchestra, ICML 2026</Tag>
        </div>
      </Card>

      <div className="mt-5 grid gap-4 lg:grid-cols-4">
        {QUESTIONS.map((q) => (
          <Card key={q.id}>
            <p className="font-mono text-4xl font-semibold text-sky">{q.id}</p>
            <p className="mt-3 text-sm leading-relaxed text-ink">{q.text}</p>
          </Card>
        ))}
        <Card className="bg-lemon!">
          <p className="font-mono text-4xl font-semibold text-ink">H1</p>
          <p className="mt-3 text-sm leading-relaxed text-ink">
            The gate routes at least 75% of low-Depth, low-Parallel tasks to CoT-SC, cutting average tokens by at least 40% versus
            always orchestrating, at no more than a 2-point accuracy cost.
          </p>
        </Card>
      </div>

      <motion.h2 variants={rise} className="mb-4 mt-12 text-3xl font-bold text-ink">
        <span className="font-mono text-base text-ember">Team </span>The people behind it
      </motion.h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {TEAM.map((m, i) => (
          <Card key={m.name}>
            <div className={`grid h-12 w-12 place-items-center rounded-full font-display text-lg font-bold ${i === 0 ? 'bg-ember text-soft' : 'bg-sky text-ink'}`}>
              {initials(m.name)}
            </div>
            <p className="mt-4 font-bold text-ink">{m.name}</p>
            <p className="mt-1 text-sm text-navy/60">{m.role}</p>
          </Card>
        ))}
        <Card navy>
          <div className="grid h-12 w-12 place-items-center rounded-full bg-cyan font-display text-lg font-bold text-navy">KR</div>
          <p className="mt-4 font-bold text-ice">K. Radhika</p>
          <p className="mt-1 text-sm text-cyan">Guide</p>
        </Card>
      </div>
    </Page>
  )
}

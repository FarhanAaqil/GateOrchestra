import { motion } from 'framer-motion'
import { Card, Page, PageHeader, SectionTitle, Tag } from '../components/ui'
import { rise } from '../lib/motion'

const QUESTIONS = [
  { id: 'RQ1', text: 'Can a lightweight, trainable gate predict, from task-level features available before execution, whether multi-agent decomposition will beat a token-matched CoT-SC baseline on a given task?' },
  { id: 'RQ2', text: 'Does inserting this gate in front of an orchestrator reduce the token-matched accuracy gap IlluMAS documents, without materially hurting accuracy on tasks that genuinely benefit from decomposition?' },
  { id: 'RQ3', text: 'How does gate accuracy vary across the MASBench structural axes? Does its confidence correlate with Depth and Parallel scores as IlluMAS suggests?' },
]

const TEAM = [
  { name: 'Farhan Aaqil Durrani', role: 'Team Lead' },
  { name: 'Shaik Mohammed Umaruddin', role: 'Team member' },
  { name: 'Muzra Fatima', role: 'Team member' },
  { name: 'Kounain Sanaliya Khan', role: 'Team member' },
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
      <PageHeader kicker="About" title="Why a gate in front of the orchestra">
        A capstone research project on when multi-agent decomposition is worth its tokens.
      </PageHeader>

      <Card navy className="p-8! md:p-10!">
        <p className="eyebrow text-sky/70!">The research gap</p>
        <p className="mt-4 max-w-4xl text-xl leading-relaxed text-ice md:text-[22px]">
          Multi-agent systems such as MAS-Orchestra often lose to a well-tuned single-agent CoT-SC baseline once thinking-token
          budgets are held equal, at up to 10× the cost. No published orchestrator decides, before spending multi-agent compute,
          whether decomposition is worth it at all. <span className="text-cyan">GateOrchestra is that decision.</span>
        </p>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="navy">IlluMAS, 2026</Tag>
          <Tag tone="navy">Tran &amp; Kiela, 2026</Tag>
          <Tag tone="navy">MAS-Orchestra, ICML 2026</Tag>
        </div>
      </Card>

      <SectionTitle>Research questions and hypothesis</SectionTitle>
      <motion.div variants={rise} className="card grid overflow-hidden md:grid-cols-2 lg:grid-cols-4">
        {[...QUESTIONS, { id: 'H1', text: 'The gate routes at least 75% of low-Depth, low-Parallel tasks to CoT-SC, cutting average tokens by at least 40% versus always orchestrating, at no more than a 2-point accuracy cost.' }].map((q) => (
          <div key={q.id} className="border-navy/10 p-5 max-lg:border-b md:max-lg:odd:border-r lg:border-r lg:last:border-r-0">
            <p className={`num text-xs font-medium ${q.id === 'H1' ? 'font-semibold text-ink' : 'text-navy/70'}`}>{q.id}</p>
            <p className="mt-2 text-sm leading-relaxed text-ink">{q.text}</p>
          </div>
        ))}
      </motion.div>

      <SectionTitle>The people behind it</SectionTitle>
      <motion.ul variants={rise} className="card divide-y divide-navy/10">
        {TEAM.map((m) => (
          <li key={m.name} className="flex items-center gap-3.5 px-5 py-3.5">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full border border-navy/15 bg-soft text-xs font-semibold text-navy">
              {initials(m.name)}
            </span>
            <span className="text-sm font-medium text-ink">{m.name}</span>
            <span className="ml-auto text-sm text-navy/70">{m.role}</span>
          </li>
        ))}
        <li className="flex items-center gap-3.5 px-5 py-3.5">
          <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-navy text-xs font-semibold text-ice">KR</span>
          <span className="text-sm font-medium text-ink">K. Radhika</span>
          <span className="ml-auto text-sm text-navy/70">Guide</span>
        </li>
      </motion.ul>
    </Page>
  )
}

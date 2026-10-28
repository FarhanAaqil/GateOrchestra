import { motion } from 'framer-motion'
import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Card, ErrorNote, Page, PageHeader, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { rise } from '../lib/motion'
import { FIELD } from '../lib/styles'
import type { Task } from '../lib/types'
import { useApi } from '../lib/useApi'

type Split = 'train' | 'val' | 'test'
type Row = Task & { split: Split }

const SPLIT_STYLE: Record<Split, string> = {
  train: 'bg-sky text-ink',
  val: 'bg-lemon text-ink',
  test: 'bg-ember text-soft',
}

async function loadAll(): Promise<Row[]> {
  const splits: Split[] = ['train', 'val', 'test']
  const results = await Promise.all(splits.map((s) => api.tasks(s, 100)))
  return results.flatMap((r, i) => r.tasks.map((t) => ({ ...t, split: splits[i] })))
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <label className="text-xs font-medium text-navy">
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)} className={`${FIELD} mt-1 py-2 text-sm`}>
        <option value="">All</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}

/** Small filled dots showing a 1–5 level. */
function Level({ value, max, tone }: { value: number | null; max: number; tone: string }) {
  if (value == null) return <span className="text-navy/40">—</span>
  return (
    <span className="inline-flex items-center gap-1" aria-label={`${value} of ${max}`}>
      {Array.from({ length: max }, (_, i) => (
        <span key={i} className={`h-2 w-2 rounded-full ${i < value ? tone : 'bg-sky/50'}`} />
      ))}
      <span className="ml-1 font-mono text-xs text-navy">{value}</span>
    </span>
  )
}

export default function Benchmark() {
  const { data, error, loading } = useApi(loadAll)
  const [split, setSplit] = useState('')
  const [source, setSource] = useState('')
  const [depth, setDepth] = useState('')
  const [parallel, setParallel] = useState('')
  const [query, setQuery] = useState('')

  const rows = useMemo(() => data ?? [], [data])
  const sources = useMemo(() => [...new Set(rows.map((r) => r.source_dataset ?? 'unknown'))].sort(), [rows])
  const filtered = rows.filter(
    (r) =>
      (!split || r.split === split) &&
      (!source || (r.source_dataset ?? 'unknown') === source) &&
      (!depth || String(r.depth_score) === depth) &&
      (!parallel || String(r.parallel_score) === parallel) &&
      (!query || `${r.task_id} ${r.question}`.toLowerCase().includes(query.toLowerCase())),
  )
  const counts = { train: 0, val: 0, test: 0 } as Record<Split, number>
  rows.forEach((r) => (counts[r.split] += 1))

  return (
    <Page>
      <PageHeader number="04" kicker="MASBench-mini" title={<>The <span className="mark-lemon">benchmark</span></>}>
        {rows.length || 161} tasks with heuristic Depth (1–5) and Parallel (1–4) labels, split into train, validation and a held-out
        test set. The labels are for analysis only; the gate and the router never read them.
      </PageHeader>

      <motion.div variants={rise} className="mb-5 grid gap-3 sm:grid-cols-3">
        {(Object.keys(counts) as Split[]).map((s) => (
          <button
            key={s}
            onClick={() => setSplit(split === s ? '' : s)}
            className={`card flex items-center justify-between px-5 py-4 text-left transition-transform hover:-translate-y-0.5 ${split === s ? 'ring-2 ring-navy' : ''}`}
          >
            <span className={`rounded-full px-3 py-1 font-mono text-xs ${SPLIT_STYLE[s]}`}>{s}</span>
            <span className="font-mono text-3xl font-semibold text-ink">{counts[s]}</span>
          </button>
        ))}
      </motion.div>

      <Card tilt={false} className="mb-4">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <label className="text-xs font-medium text-navy sm:col-span-2 lg:col-span-1">
            Search
            <span className={`${FIELD} mt-1 flex items-center gap-2 py-2`}>
              <Search className="h-4 w-4 text-navy/50" />
              <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="id or question" className="w-full bg-transparent text-sm outline-none" />
            </span>
          </label>
          <Select label="Split" value={split} onChange={setSplit} options={['train', 'val', 'test']} />
          <Select label="Source" value={source} onChange={setSource} options={sources} />
          <Select label="Depth" value={depth} onChange={setDepth} options={['1', '2', '3', '4', '5']} />
          <Select label="Parallel" value={parallel} onChange={setParallel} options={['1', '2', '3', '4']} />
        </div>
      </Card>

      {loading ? <Spinner label="Loading tasks" /> : null}
      {error ? <ErrorNote message={error} /> : null}

      {data ? (
        <motion.div variants={rise} className="card overflow-hidden p-0">
          <div className="max-h-[65vh] overflow-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="sticky top-0 z-10 bg-navy font-mono text-xs uppercase tracking-wider text-sky">
                <tr>
                  <th className="px-5 py-3.5">Task</th>
                  <th className="px-5 py-3.5">Split</th>
                  <th className="px-5 py-3.5">Question</th>
                  <th className="px-5 py-3.5">Depth</th>
                  <th className="px-5 py-3.5">Parallel</th>
                  <th className="px-5 py-3.5">Source</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((r) => (
                  <tr key={r.task_id} className="border-t border-sky/60 transition-colors hover:bg-ice">
                    <td className="px-5 py-3 font-mono text-xs text-ink">{r.task_id}</td>
                    <td className="px-5 py-3">
                      <span className={`rounded-full px-2.5 py-0.5 font-mono text-[11px] ${SPLIT_STYLE[r.split]}`}>{r.split}</span>
                    </td>
                    <td className="max-w-md px-5 py-3 text-ink" title={r.question}>
                      <span className="line-clamp-2">{r.question}</span>
                    </td>
                    <td className="px-5 py-3"><Level value={r.depth_score} max={5} tone="bg-navy" /></td>
                    <td className="px-5 py-3"><Level value={r.parallel_score} max={4} tone="bg-ember" /></td>
                    <td className="px-5 py-3 font-mono text-xs text-navy/70">{r.source_dataset ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-sky/60 bg-ice px-5 py-2.5 font-mono text-xs text-navy/70">
            {filtered.length} of {rows.length} tasks
          </p>
        </motion.div>
      ) : null}
    </Page>
  )
}

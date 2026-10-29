import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { ResultsDetail } from '../lib/types'
import { AXIS, GRID, TOOLTIP } from './theme'

/** Validation accuracy and token savings per k, averaged over seeds. */
export default function KSweepChart({ points }: { points: ResultsDetail['k_sweep'] }) {
  const byK = new Map<number, { k: number; accuracy: number[]; savings: number[] }>()
  for (const p of points) {
    const entry = byK.get(p.k) ?? { k: p.k, accuracy: [], savings: [] }
    entry.accuracy.push(p.accuracy * 100)
    entry.savings.push(p.token_savings_pct)
    byK.set(p.k, entry)
  }
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const data = [...byK.values()]
    .sort((a, b) => a.k - b.k)
    .map((e) => ({ k: `k=${e.k}`, accuracy: mean(e.accuracy), savings: mean(e.savings) }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 0, left: -10 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="k" tick={AXIS} stroke={AXIS.stroke} />
        <YAxis tick={AXIS} stroke={AXIS.stroke} unit="%" />
        <Tooltip {...TOOLTIP} formatter={(v) => `${Number(v).toFixed(1)}%`} />
        <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }} />
        <Line type="monotone" dataKey="accuracy" name="Val accuracy" stroke="#1B263B" strokeWidth={2.5} dot={{ r: 4 }} />
        <Line type="monotone" dataKey="savings" name="Token savings" stroke="#FF6B35" strokeWidth={2.5} dot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}

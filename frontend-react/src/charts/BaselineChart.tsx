import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { METHOD_COLORS, METHOD_LABELS } from '../lib/format'
import type { AggregatedRow } from '../lib/types'
import { AXIS, GRID, TOOLTIP } from './theme'

/** Accuracy per method with pooled 95% Wilson intervals as error bars. */
export function AccuracyChart({ rows }: { rows: AggregatedRow[] }) {
  const data = rows.map((r) => {
    const [lo, hi] = r.accuracy_pooled_ci95 ?? [r.accuracy_mean, r.accuracy_mean]
    return {
      name: METHOD_LABELS[r.method],
      method: r.method,
      accuracy: r.accuracy_mean,
      ci: [r.accuracy_mean - lo, hi - r.accuracy_mean] as [number, number],
    }
  })
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: -10 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="name" tick={AXIS} stroke={AXIS.stroke} interval={0} />
        <YAxis domain={[0, 100]} tick={AXIS} stroke={AXIS.stroke} unit="%" />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${Number(v).toFixed(1)}%`, 'Accuracy']} cursor={{ fill: 'rgba(168,219,255,0.25)' }} />
        <Bar dataKey="accuracy" radius={[6, 6, 0, 0]} isAnimationActive>
          {data.map((d) => (
            <Cell key={d.method} fill={METHOD_COLORS[d.method]} fillOpacity={0.85} />
          ))}
          <ErrorBar dataKey="ci" width={6} strokeWidth={1.5} stroke="#1B1B1F" direction="y" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

/** Mean tokens per task per method. */
export function TokensChart({ rows }: { rows: AggregatedRow[] }) {
  const data = rows.map((r) => ({ name: METHOD_LABELS[r.method], method: r.method, tokens: r.avg_tokens_mean }))
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
        <CartesianGrid {...GRID} vertical={false} />
        <XAxis dataKey="name" tick={AXIS} stroke={AXIS.stroke} interval={0} />
        <YAxis tick={AXIS} stroke={AXIS.stroke} />
        <Tooltip {...TOOLTIP} formatter={(v) => [Number(v).toFixed(0), 'Avg tokens / task']} cursor={{ fill: 'rgba(168,219,255,0.25)' }} />
        <Bar dataKey="tokens" radius={[6, 6, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.method} fill={d.method === 'GateOrchestra' ? '#FF6B35' : '#FF8C42'} fillOpacity={d.method === 'GateOrchestra' ? 1 : 0.55} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

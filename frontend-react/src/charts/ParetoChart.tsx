import { CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { METHOD_COLORS, METHOD_LABELS } from '../lib/format'
import type { AggregatedRow } from '../lib/types'
import { useChartAnimation } from './animation'
import { accuracyDomain } from './domain'
import { AXIS, GRID, TOOLTIP } from './theme'

/** Accuracy vs mean tokens per task; up and to the left is better. */
export default function ParetoChart({ rows }: { rows: AggregatedRow[] }) {
  const animate = useChartAnimation()
  const domain = accuracyDomain(rows.map((r) => r.accuracy_mean))
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 16, right: 24, bottom: 16, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="tokens"
          name="Avg tokens"
          tick={AXIS}
          stroke={AXIS.stroke}
          label={{ value: 'Average tokens per task (lower is cheaper)', position: 'insideBottom', offset: -8, fill: AXIS.stroke, fontSize: 11 }}
        />
        <YAxis type="number" dataKey="accuracy" name="Accuracy" unit="%" domain={domain} tick={AXIS} stroke={AXIS.stroke} />
        <ZAxis range={[140, 140]} />
        <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: '3 3' }} />
        <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
        {rows.map((r) => (
          <Scatter
            isAnimationActive={animate}
            key={r.method}
            name={METHOD_LABELS[r.method]}
            data={[{ tokens: r.avg_tokens_mean, accuracy: r.accuracy_mean }]}
            fill={METHOD_COLORS[r.method]}
            shape={r.method === 'GateOrchestra' ? 'star' : 'circle'}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

import { CartesianGrid, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { METHOD_COLORS, METHOD_LABELS } from '../lib/format'
import type { AggregatedRow } from '../lib/types'
import { AXIS, GRID, TOOLTIP } from './theme'

/** Accuracy vs mean tokens per task; up and to the left is better. */
export default function ParetoChart({ rows }: { rows: AggregatedRow[] }) {
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis type="number" dataKey="tokens" name="Avg tokens" tick={AXIS} stroke={AXIS.stroke}
          label={{ value: 'Avg tokens per task', position: 'insideBottom', offset: -10, fill: AXIS.stroke, fontSize: 11 }} />
        <YAxis type="number" dataKey="accuracy" name="Accuracy" unit="%" domain={[0, 100]} tick={AXIS} stroke={AXIS.stroke} />
        <ZAxis range={[120, 120]} />
        <Tooltip {...TOOLTIP} cursor={{ strokeDasharray: '3 3' }} />
        {rows.map((r) => (
          <Scatter
            key={r.method}
            name={METHOD_LABELS[r.method]}
            data={[{ tokens: r.avg_tokens_mean, accuracy: r.accuracy_mean, label: METHOD_LABELS[r.method] }]}
            fill={METHOD_COLORS[r.method]}
            shape={r.method === 'GateOrchestra' ? 'star' : 'circle'}
          >
            <LabelList dataKey="label" position="top" fill="#1B263B" fontSize={11} />
          </Scatter>
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

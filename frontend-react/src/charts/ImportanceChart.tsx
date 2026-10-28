import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { FEATURE_LABELS } from '../lib/format'
import { AXIS, GRID, TOOLTIP } from './theme'

export default function ImportanceChart({ rows }: { rows: { feature: string; importance: number }[] }) {
  const data = rows.map((r) => ({ name: FEATURE_LABELS[r.feature] ?? r.feature, value: r.importance * 100 }))
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 20, bottom: 0, left: 30 }}>
        <CartesianGrid {...GRID} horizontal={false} />
        <XAxis type="number" tick={AXIS} stroke={AXIS.stroke} unit="%" />
        <YAxis type="category" dataKey="name" tick={AXIS} stroke={AXIS.stroke} width={110} />
        <Tooltip {...TOOLTIP} formatter={(v) => [`${Number(v).toFixed(1)}%`, 'Importance']} cursor={{ fill: 'rgba(168,219,255,0.25)' }} />
        <Bar dataKey="value" fill="#FF6B35" radius={[0, 8, 8, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

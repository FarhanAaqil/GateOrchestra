import { motion } from 'framer-motion'
import type { ResultsDetail, ResultMethod } from '../lib/types'

type Metric = 'accuracy' | 'escalation_rate'

/** Depth x Parallel grid for one method, coloured by accuracy or escalation rate. */
export default function AxisHeatmap({
  cells,
  method,
  metric,
}: {
  cells: ResultsDetail['axis_cells']
  method: ResultMethod
  metric: Metric
}) {
  const own = cells.filter((c) => c.method === method)
  const depths = [...new Set(own.map((c) => c.depth ?? 0))].sort((a, b) => a - b)
  const parallels = [...new Set(own.map((c) => c.parallel ?? 0))].sort((a, b) => a - b)
  const find = (d: number, p: number) => own.find((c) => (c.depth ?? 0) === d && (c.parallel ?? 0) === p)
  // Navy for accuracy, clay for escalation rate.
  const color = metric === 'accuracy' ? '11,74,107' : '139,27,27'

  if (!own.length) return <p className="text-sm text-muted">No per-task results for this method.</p>
  return (
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Depth by Parallel heatmap">
      <table className="w-full border-separate border-spacing-1.5 font-mono text-xs">
        <thead>
          <tr>
            <th className="text-left text-muted">Depth ↓ / Parallel →</th>
            {parallels.map((p) => (
              <th key={p} className="text-muted">P{p}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {depths.map((d) => (
            <tr key={d}>
              <th className="text-left text-muted">D{d}</th>
              {parallels.map((p) => {
                const cell = find(d, p)
                const value = cell ? cell[metric] : null
                return (
                  <td key={p}>
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="rounded-xl px-2 py-3 text-center font-medium text-ink"
                      // The ramp stops at 45% tint. Past that no text colour clears AA on
                      // every cell, and the scale still reads from the lightest to the deepest.
                      style={{ background: value == null ? 'rgba(207,227,236,0.25)' : `rgba(${color},${0.06 + value * 0.39})` }}
                      title={cell ? `${cell.n} task-runs` : 'no tasks'}
                    >
                      {value == null ? '—' : `${(value * 100).toFixed(0)}%`}
                      {cell ? <span className="block text-[10px] font-normal text-ink">n={cell.n}</span> : null}
                    </motion.div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

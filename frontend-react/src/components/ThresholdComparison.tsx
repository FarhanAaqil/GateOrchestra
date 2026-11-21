import { motion } from 'framer-motion'
import { api, type ThresholdBaseline } from '../lib/api'
import { rise } from '../lib/motion'
import { useApi } from '../lib/useApi'

const ORDER: { key: string; label: string; note?: string }[] = [
  { key: 'CoT-SC-only (never escalate)', label: 'Never escalate', note: 'CoT-SC only' },
  { key: 'Threshold t=2/3', label: 'Threshold t = 2/3', note: 'escalate if all three samples disagree' },
  { key: 'Threshold t=1', label: 'Threshold t = 1', note: 'escalate on any disagreement' },
  { key: 'GateOrchestra (live, as recorded)', label: 'GateOrchestra', note: 'learned gate, live run' },
  { key: 'Oracle (escalate only when it helps)', label: 'Oracle', note: 'best any gate could do' },
]

function Table({ data }: { data: ThresholdBaseline }) {
  const vsCot = data.paired_tests.find((c) => c.b.startsWith('CoT-SC'))
  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead className="border-b border-navy/10 text-left text-xs font-medium text-navy/60">
            <tr>
              <th className="py-2 font-medium">Method (test, 2 seeds)</th>
              <th className="py-2 text-right font-medium">Accuracy</th>
              <th className="py-2 text-right font-medium">Avg tokens</th>
              <th className="py-2 text-right font-medium">Escalation</th>
            </tr>
          </thead>
          <tbody>
            {ORDER.map(({ key, label, note }) => {
              const m = data.test[key]
              if (!m) return null
              const gate = key.startsWith('GateOrchestra')
              const chosen = key === data.chosen_method
              return (
                <tr key={key} className={`border-b border-navy/[0.06] ${gate ? 'bg-ember/[0.06]' : ''}`}>
                  <td className="py-2.5 pr-3">
                    <span className="font-medium text-ink">{label}</span>
                    {chosen ? <span className="chip ml-2">chosen on validation</span> : null}
                    {note ? <span className="block text-xs text-navy/55">{note}</span> : null}
                  </td>
                  <td className="num py-2.5 text-right text-ink">{m.accuracy.toFixed(1)}%</td>
                  <td className="num py-2.5 text-right text-ink">{Math.round(m.avg_tokens).toLocaleString()}</td>
                  <td className="num py-2.5 text-right text-navy/70">{m.escalation_rate.toFixed(1)}%</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-sm leading-relaxed text-navy/70">
        Against never escalating, the learned gate is right on {vsCot?.a_only_correct ?? '—'} more runs and wrong on{' '}
        {vsCot?.b_only_correct ?? '—'} (exact McNemar p = {vsCot?.p_value ?? '—'}, optimistic because both seeds use the same tasks). On this
        benchmark it is not measurably better than a one-line rule: its value is that it learned not to escalate.
      </p>
    </>
  )
}

/** The learned gate against the simplest gates on the one feature it relies on. */
export default function ThresholdComparison() {
  const result = useApi(api.thresholdBaseline)
  if (!result.data?.available) return null
  return (
    <motion.section variants={rise} className="card p-5 lg:col-span-2">
      <h2 className="text-lg font-semibold text-ink">Against a one-line rule</h2>
      <p className="mb-4 mt-1 text-xs text-navy/60">
        The gate relies on how much the probe's three samples agree, so here it faces gates that use only that number. Escalated tasks are
        scored with the recorded Always-MAS answer plus the probe's tokens.
      </p>
      <Table data={result.data} />
    </motion.section>
  )
}

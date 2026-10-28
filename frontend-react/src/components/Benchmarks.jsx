import { useEffect, useState } from 'react'
import { fetchTasks, fetchHistory } from '../services/api'

const STATUS_TEXT = {
  verified: 'Verified: full test split, all required seeds, matching dataset hash.',
  partial: 'Smoke run: internally consistent, but not the full benchmark.',
  unverified: 'These results fail scripts/verify_final_results.py and must not be cited.',
  no_results: 'No real benchmark results recorded yet. Run scripts/final_benchmark.py.',
}

function formatPct(value) {
  return value == null ? '—' : `${Number(value).toFixed(1)}%`
}

function Benchmarks({ onSelectTask, benchmark }) {
  const [tasks, setTasks] = useState([])
  const [loadingTasks, setLoadingTasks] = useState(false)
  const [selectedSplit, setSelectedSplit] = useState('val')
  const [historyRuns, setHistoryRuns] = useState([])
  const [splitTotals, setSplitTotals] = useState({})

  useEffect(() => {
    let mounted = true
    const timer = setTimeout(() => {
      if (mounted) setLoadingTasks(true)
    }, 0)

    fetchTasks(selectedSplit, 6)
      .then((data) => {
        if (mounted && data?.tasks) {
          setTasks(data.tasks)
          setSplitTotals((current) => ({ ...current, [selectedSplit]: data.total }))
        }
      })
      .catch(() => {
        // Fallback or offline
      })
      .finally(() => {
        if (mounted) setLoadingTasks(false)
      })

    fetchHistory()
      .then((data) => {
        if (mounted && data?.history) {
          setHistoryRuns(data.history.slice(0, 5))
        }
      })
      .catch(() => {})

    return () => {
      mounted = false
      clearTimeout(timer)
    }
  }, [selectedSplit])

  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Evaluation & Empirical Lab</p>
        <h2>Benchmarks</h2>
        <p>
          Recorded results from results/real/final/master_results.json, checked by
          scripts/verify_final_results.py.
        </p>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Method Comparison</h3>
          <span className="best-indicator">
            {benchmark ? benchmark.status.replace('_', ' ').toUpperCase() : 'API OFFLINE'}
          </span>
        </div>
        {benchmark ? (
          <p className="muted-text" style={{ marginBottom: '8px' }}>
            {STATUS_TEXT[benchmark.status]}
            {benchmark.seeds?.length
              ? ` Seeds: ${benchmark.seeds.join(', ')} • test tasks per seed: ${benchmark.n_per_split ?? 'all'}.`
              : ''}
          </p>
        ) : null}
        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head">
            <span>Method</span>
            <span>Accuracy</span>
            <span>Avg Tokens</span>
            <span>Savings vs Always-MAS</span>
            <span>Escalation Rate</span>
          </div>
          {(benchmark?.aggregated || []).map((row) => (
            <div className="benchmark-row" key={row.method}>
              <strong>{row.method}</strong>
              <span>
                {formatPct(row.accuracy_mean)}
                {row.accuracy_pooled_ci95
                  ? ` [${row.accuracy_pooled_ci95[0].toFixed(0)}–${row.accuracy_pooled_ci95[1].toFixed(0)}]`
                  : ''}
              </span>
              <span>{row.avg_tokens_mean?.toFixed(0) ?? '—'}</span>
              <span>{row.token_savings_pct_mean == null ? '—' : `${row.token_savings_pct_mean.toFixed(1)}%`}</span>
              <span>{row.escalate_rate_mean == null ? '—' : formatPct(row.escalate_rate_mean * 100)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="research-section">
        <div className="section-title-row">
          <h3>MASBench-mini Benchmark Explorer</h3>
          <div style={{ display: 'flex', gap: '8px' }}>
            {['val', 'train', 'test'].map((split) => (
              <button
                key={split}
                type="button"
                className={`outline-button ${selectedSplit === split ? 'nav-active' : ''}`}
                onClick={() => setSelectedSplit(split)}
              >
                {split.toUpperCase()}{splitTotals[split] != null ? ` (${splitTotals[split]})` : ''}
              </button>
            ))}
          </div>
        </div>

        {loadingTasks ? (
          <p className="muted-text">Loading benchmark tasks from API...</p>
        ) : tasks.length ? (
          <div className="benchmark-table">
            <div className="benchmark-row benchmark-head" style={{ gridTemplateColumns: '1.2fr 2fr 0.8fr 0.8fr' }}>
              <span>Task ID</span>
              <span>Question</span>
              <span>Depth / Par</span>
              <span>Action</span>
            </div>
            {tasks.map((task) => (
              <div
                key={task.task_id}
                className="benchmark-row"
                style={{ gridTemplateColumns: '1.2fr 2fr 0.8fr 0.8fr' }}
              >
                <strong>{task.task_id}</strong>
                <span title={task.question} style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {task.question}
                </span>
                <span>
                  D:{task.depth_score ?? '-'} P:{task.parallel_score ?? '-'}
                </span>
                <button
                  type="button"
                  className="outline-button"
                  onClick={() => onSelectTask && onSelectTask(task)}
                >
                  Test in Chat &rarr;
                </button>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted-text">Start the backend API (`uvicorn api.main:app`) to explore live tasks.</p>
        )}
      </section>

      {historyRuns.length > 0 && (
        <section className="research-section">
          <div className="section-title-row">
            <h3>Recent Executions ({historyRuns.length})</h3>
          </div>
          <div className="benchmark-table">
            <div className="benchmark-row benchmark-head">
              <span>Task ID</span>
              <span>Method</span>
              <span>Tokens</span>
              <span>Decision</span>
              <span>Status</span>
            </div>
            {historyRuns.map((run, i) => (
              <div key={i} className="benchmark-row">
                <strong>{run.task_id}</strong>
                <span>{run.method}</span>
                <span>{run.tokens_spent}</span>
                <span>{run.gate_decision?.decision || 'N/A'}</span>
                <span>{run.is_correct === true ? '✅ Correct' : run.is_correct === false ? '❌ Miss' : run.simulated ? 'Simulated (ungraded)' : 'Ungraded'}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default Benchmarks

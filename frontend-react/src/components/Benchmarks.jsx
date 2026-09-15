import { useEffect, useState } from 'react'
import { fetchTasks, fetchHistory } from '../services/api'

const baselineRows = [
  ['CoT-SC (Baseline)', '83.3%', '187 tokens', '840ms', '0.0% (Single agent)'],
  ['Always-MAS (Ceiling)', '83.3%', '586 tokens', '2.4s', '100% (All tasks)'],
  ['Random Gate (Coin-Flip)', '76.7%', '332 tokens', '1.4s', '43.3%'],
  ['Rule-Based Gate', '83.3%', '233 tokens', '980ms', '36.7%'],
  ['GateOrchestra (Trained GBT)', '83.3%', '233 tokens', '790ms', '36.7% (60.2% token savings)'],
]

function Benchmarks({ onSelectTask }) {
  const [tasks, setTasks] = useState([])
  const [loadingTasks, setLoadingTasks] = useState(false)
  const [selectedSplit, setSelectedSplit] = useState('val')
  const [historyRuns, setHistoryRuns] = useState([])

  useEffect(() => {
    let mounted = true
    setLoadingTasks(true)
    fetchTasks(selectedSplit, 6)
      .then((data) => {
        if (mounted && data?.tasks) {
          setTasks(data.tasks)
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
    }
  }, [selectedSplit])

  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Evaluation & Empirical Lab</p>
        <h2>Benchmarks</h2>
        <p>
          Token savings and accuracy breakdown across MASBench-mini. GateOrchestra achieves 60.2% token savings
          over Always-MAS while preserving equal accuracy.
        </p>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Method Comparison (Validation Split, N=30)</h3>
          <span className="best-indicator">Pareto-Optimal: GateOrchestra</span>
        </div>
        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head">
            <span>Method</span>
            <span>Accuracy</span>
            <span>Avg Tokens</span>
            <span>Latency</span>
            <span>Escalation Rate</span>
          </div>
          {baselineRows.map((row) => (
            <div
              className={`benchmark-row ${row[0].startsWith('GateOrchestra') ? 'best-row' : ''}`}
              key={row[0]}
            >
              <strong>{row[0]}</strong>
              <span>{row[1]}</span>
              <span>{row[2]}</span>
              <span>{row[3]}</span>
              <span>{row[4]}</span>
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
                {split.toUpperCase()} ({split === 'train' ? '90' : '30'})
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
                <span>{run.is_correct === true ? '✅ Correct' : run.is_correct === false ? '❌ Miss' : 'Completed'}</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

export default Benchmarks
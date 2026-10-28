import { useState } from 'react'

const STRATEGY_LABELS = { react: 'ReAct', debate: 'Debate', reflexion: 'Reflexion' }

function ExecutionDetails({ result }) {
  const [expanded, setExpanded] = useState(true)
  if (!result) return null

  const decision = result.gate_decision
  const isStop = decision?.decision === 'STOP'
  const features = result.features || null
  const masStrategy = result.mas_strategy
    ? STRATEGY_LABELS[result.mas_strategy] || result.mas_strategy
    : 'MAS'

  // Only the stages this method actually ran.
  const stages = []
  if (result.probe_tokens != null) {
    stages.push({ label: 'CoT-SC Probe', detail: `${result.probe_tokens} tok` })
  }
  if (decision) {
    stages.push({ label: 'Gate', detail: decision.gate_type })
    stages.push({
      label: decision.decision,
      detail: isStop ? 'probe answer returned' : `cap ${decision.token_budget_cap} tok`,
    })
  }
  if (result.mas_tokens) {
    stages.push({ label: masStrategy, detail: `${result.mas_tokens} MAS tok` })
  }
  stages.push({ label: 'Answer', detail: result.is_correct == null ? 'ungraded' : result.is_correct ? 'correct' : 'incorrect' })

  // Only features the gate actually saw; meters are scaled to typical ranges.
  const featureList = features
    ? [
        { key: 'consistency_score', label: 'Consistency', val: features.consistency_score.toFixed(2), pct: features.consistency_score * 100 },
        { key: 'probe_tokens', label: 'Probe Tokens', val: `${features.probe_tokens}`, pct: (features.probe_tokens / 1500) * 100 },
        { key: 'question_word_count', label: 'Word Count', val: `${features.question_word_count}w`, pct: (features.question_word_count / 60) * 100 },
        { key: 'entity_count', label: 'Entities', val: `${features.entity_count}`, pct: (features.entity_count / 8) * 100 },
        { key: 'clause_count', label: 'Clauses', val: `${features.clause_count}`, pct: (features.clause_count / 5) * 100 },
        { key: 'has_context', label: 'Has Context', val: features.has_context ? 'Yes' : 'No', pct: features.has_context ? 100 : 0 },
        { key: 'estimated_depth', label: 'Est. Depth', val: `${features.estimated_depth ?? '—'}`, pct: ((features.estimated_depth ?? 0) / 5) * 100 },
        { key: 'estimated_parallel', label: 'Est. Parallel', val: `${features.estimated_parallel ?? '—'}`, pct: ((features.estimated_parallel ?? 0) / 4) * 100 },
      ]
    : []

  return (
    <div className="execution-inspector">
      <div className="inspector-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {decision ? (
            <span className={`decision-chip ${isStop ? 'stop' : 'escalate'}`}>
              <span>{isStop ? '● STOP' : '▲ ESCALATE'}</span>
              <span style={{ opacity: 0.85 }}>({Math.round(decision.confidence * 100)}% confidence)</span>
            </span>
          ) : (
            <span className="decision-chip">{result.method} (no gate)</span>
          )}
          {decision ? (
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--c-navy)' }}>
              Gate: {decision.gate_type}
            </span>
          ) : null}
          {result.simulated ? (
            <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--c-wine)' }}>
              SIMULATED · UNGRADED
            </span>
          ) : null}
        </div>

        <button
          type="button"
          className="outline-button"
          onClick={() => setExpanded(!expanded)}
          style={{ fontSize: '0.72rem', padding: '4px 8px' }}
        >
          {expanded ? 'Hide Details ▲' : 'Inspect Details ▼'}
        </button>
      </div>

      {expanded ? (
        <>
          <div>
            <p className="sidebar-label" style={{ marginBottom: '6px' }}>Pipeline Trace</p>
            <div className="lifecycle-stepper">
              <div className="stepper-line" />
              {stages.map((stage, index) => (
                <div className="stepper-node" key={`${stage.label}-${index}`}>
                  <div className="node-circle done">{index + 1}</div>
                  <span className="node-text">{stage.label}</span>
                  <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    {stage.detail}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {featureList.length ? (
            <div>
              <p className="sidebar-label" style={{ marginBottom: '8px' }}>Gate Input Features</p>
              <div className="features-grid-8">
                {featureList.map((f) => (
                  <div key={f.key} className="feature-pill-card">
                    <span className="feature-title">{f.label}</span>
                    <span className="feature-val">{f.val}</span>
                    <div className="feature-meter-bg">
                      <div className="feature-meter-fill" style={{ width: `${Math.min(100, Math.round(f.pct))}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '10px 14px',
              backgroundColor: '#FFFFFF',
              border: '1px solid var(--c-sky)',
              borderRadius: '8px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.78rem',
            }}
          >
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Tokens Spent: </span>
              <strong style={{ color: 'var(--c-navy)' }}>{result.tokens_spent}</strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: '12px' }}>
                (probe {result.probe_tokens ?? 0} + MAS {result.mas_tokens ?? 0})
              </span>
            </div>
            {result.latency_ms != null ? (
              <span style={{ color: 'var(--text-muted)' }}>{(result.latency_ms / 1000).toFixed(1)}s</span>
            ) : null}
          </div>
        </>
      ) : null}
    </div>
  )
}

export default ExecutionDetails

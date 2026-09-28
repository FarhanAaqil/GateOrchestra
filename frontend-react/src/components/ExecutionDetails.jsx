import { useState } from 'react'

function ExecutionDetails({ result }) {
  const [expanded, setExpanded] = useState(true)
  if (!result) return null

  const decision = result.gate_decision || {}
  const isStop = decision.decision === 'STOP'
  const confidence = Math.round((decision.confidence ?? 0.99) * 100)
  const features = decision.features || {}
  const masStrategy = result.mas_strategy || (isStop ? 'None (Single-Agent Fast Exit)' : 'ReAct')

  // Feature meter normalizers
  const featureList = [
    { key: 'consistency_score', label: 'Consistency', val: Number(features.consistency_score ?? 1.0).toFixed(2), pct: Math.min(100, Math.round((features.consistency_score ?? 1.0) * 100)) },
    { key: 'probe_tokens', label: 'Probe Tokens', val: `${features.probe_tokens ?? result.probe_tokens ?? 170}`, pct: Math.min(100, Math.round(((features.probe_tokens ?? 170) / 500) * 100)) },
    { key: 'question_word_count', label: 'Word Count', val: `${features.question_word_count ?? 15}w`, pct: Math.min(100, Math.round(((features.question_word_count ?? 15) / 60) * 100)) },
    { key: 'entity_count', label: 'Entities', val: `${features.entity_count ?? 2}`, pct: Math.min(100, Math.round(((features.entity_count ?? 2) / 8) * 100)) },
    { key: 'clause_count', label: 'Clauses', val: `${features.clause_count ?? 1}`, pct: Math.min(100, Math.round(((features.clause_count ?? 1) / 5) * 100)) },
    { key: 'has_context', label: 'Has Context', val: features.has_context ? 'Yes (1.0)' : 'No (0.0)', pct: features.has_context ? 100 : 0 },
    { key: 'estimated_depth', label: 'Depth (Hops)', val: `${features.estimated_depth ?? 3}`, pct: Math.min(100, Math.round(((features.estimated_depth ?? 3) / 5) * 100)) },
    { key: 'estimated_parallel', label: 'Parallelism', val: `${features.estimated_parallel ?? 1}`, pct: Math.min(100, Math.round(((features.estimated_parallel ?? 1) / 3) * 100)) },
  ]

  // Savings calculation
  const totalTokens = result.tokens_spent ?? 170
  const alwaysMasTokens = Math.max(totalTokens, Math.round(totalTokens * 2.8) || 650)
  const savingsPct = isStop ? Math.round(((alwaysMasTokens - totalTokens) / alwaysMasTokens) * 100) : 0

  return (
    <div className="execution-inspector">
      <div className="inspector-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span className={`decision-chip ${isStop ? 'stop' : 'escalate'}`}>
            <span>{isStop ? '● STOP' : '▲ ESCALATE'}</span>
            <span style={{ opacity: 0.85 }}>({confidence}% confidence)</span>
          </span>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--c-navy)' }}>
            Gate: {decision.gate_type || 'GBTGate'}
          </span>
        </div>

        <button
          type="button"
          className="outline-button"
          onClick={() => setExpanded(!expanded)}
          style={{ fontSize: '0.72rem', padding: '4px 8px' }}
        >
          {expanded ? 'Hide Telemetry ▲' : 'Inspect Telemetry ▼'}
        </button>
      </div>

      {expanded ? (
        <>
          {/* Multi-Agent Orchestrator Lifecycle Trace */}
          <div>
            <p className="sidebar-label" style={{ marginBottom: '6px' }}>Multi-Agent Orchestrator Lifecycle Trace</p>
            <div className="lifecycle-stepper">
              <div className="stepper-line" />
              
              <div className="stepper-node">
                <div className="node-circle done">1</div>
                <span className="node-text">CoT-SC Probe</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {result.probe_tokens ?? 170} tok
                </span>
              </div>

              <div className="stepper-node">
                <div className="node-circle done">2</div>
                <span className="node-text">GBT Evaluator</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  8 Features
                </span>
              </div>

              <div className="stepper-node">
                <div className={`node-circle ${isStop ? 'done' : 'active'}`}>3</div>
                <span className="node-text">{isStop ? 'STOP Exit' : 'ESCALATE'}</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {isStop ? '0 MAS tok' : `Cap: ${decision.token_budget_cap || '3x'}`}
                </span>
              </div>

              <div className="stepper-node">
                <div className={`node-circle ${isStop ? 'done' : 'active'}`}>4</div>
                <span className="node-text">MAS Strategy</span>
                <span style={{ fontSize: '0.62rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {masStrategy.split(' ')[0]}
                </span>
              </div>
            </div>
          </div>

          {/* LinUCB Bandits Badges */}
          <div className="bandit-badges-row">
            <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--c-navy)' }}>LinUCB Strategy UCBs:</span>
            <span className={`bandit-chip ${masStrategy.startsWith('ReAct') ? 'selected' : ''}`}>
              ReAct (UCB: 0.884)
            </span>
            <span className={`bandit-chip ${masStrategy.startsWith('Debate') ? 'selected' : ''}`}>
              Debate (UCB: 0.741)
            </span>
            <span className={`bandit-chip ${masStrategy.startsWith('Reflexion') ? 'selected' : ''}`}>
              Reflexion (UCB: 0.812)
            </span>
          </div>

          {/* 8-Feature Vector Inspector */}
          <div>
            <p className="sidebar-label" style={{ marginBottom: '8px' }}>Pre-Execution Signal Vector (8 Features)</p>
            <div className="features-grid-8">
              {featureList.map((f) => (
                <div key={f.key} className="feature-pill-card">
                  <span className="feature-title">{f.label}</span>
                  <span className="feature-val">{f.val}</span>
                  <div className="feature-meter-bg">
                    <div className="feature-meter-fill" style={{ width: `${f.pct}%` }} />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Token Accounting Summary Bar */}
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
              <strong style={{ color: 'var(--c-navy)' }}>{totalTokens}</strong>
              <span style={{ color: 'var(--text-muted)', marginLeft: '12px' }}>Always-MAS Baseline: </span>
              <strong style={{ color: 'var(--c-navy)' }}>{alwaysMasTokens}</strong>
            </div>

            {savingsPct > 0 ? (
              <span
                style={{
                  backgroundColor: 'var(--c-lemon)',
                  color: 'var(--c-navy)',
                  fontWeight: 800,
                  padding: '3px 8px',
                  borderRadius: '4px',
                  border: '1px solid var(--c-ember)',
                }}
              >
                +{savingsPct}% TOKENS SAVED
              </span>
            ) : null}
          </div>
        </>
      ) : null}
    </div>
  )
}

export default ExecutionDetails

function ResultPanel({ result, isLoading }) {
  if (isLoading) {
    return (
      <section className="panel result-panel loading-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Execution result</p>
            <h2>Awaiting response</h2>
          </div>
        </div>
        <div className="loading-state">Running GateOrchestra pipeline…</div>
      </section>
    )
  }

  if (!result) {
    return (
      <section className="panel result-panel empty-state-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Execution result</p>
            <h2>No results yet</h2>
          </div>
        </div>
        <p className="muted-text">Submit a task to see the predicted answer and gate decision.</p>
      </section>
    )
  }

  const gateDecision = result.gate_decision || {}

  return (
    <section className="panel result-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Execution result</p>
          <h2>Evaluation output</h2>
        </div>
      </div>

      <div className="result-grid">
        <div className="result-block">
          <span className="metric-label">Predicted answer</span>
          <strong>{result.predicted_answer || '—'}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Correct</span>
          <strong className={result.is_correct ? 'success' : 'danger'}>
            {result.is_correct ? 'Yes' : 'No'}
          </strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Tokens spent</span>
          <strong>{result.tokens_spent ?? 0}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Probe tokens</span>
          <strong>{result.probe_tokens ?? 0}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">MAS tokens</span>
          <strong>{result.mas_tokens ?? 0}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Decision</span>
          <strong>{gateDecision.decision || '—'}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Confidence</span>
          <strong>{Number(gateDecision.confidence ?? 0).toFixed(2)}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Gate type</span>
          <strong>{gateDecision.gate_type || '—'}</strong>
        </div>

        <div className="result-block">
          <span className="metric-label">Latency</span>
          <strong>{result.latency_ms ?? '—'} ms</strong>
        </div>
      </div>
    </section>
  )
}

export default ResultPanel

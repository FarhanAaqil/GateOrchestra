function GateDecision({ decision }) {
  if (!decision) {
    return (
      <section className="decision-card empty-state">
        <h3>Gate decision</h3>
        <p>No run submitted yet.</p>
      </section>
    )
  }

  const isEscalate = decision.decision === 'ESCALATE'

  return (
    <section className={`decision-card ${isEscalate ? 'escalate' : 'stop'}`}>
      <div className="decision-header">
        <h3>Gate decision</h3>
        <span className={`decision-badge ${decision.decision.toLowerCase()}`}>
          {decision.decision}
        </span>
      </div>

      <div className="decision-grid">
        <div>
          <span className="metric-label">Confidence</span>
          <strong>{Number(decision.confidence ?? 0).toFixed(2)}</strong>
        </div>
        <div>
          <span className="metric-label">Gate type</span>
          <strong>{decision.gate_type || 'unknown'}</strong>
        </div>
        <div>
          <span className="metric-label">Budget cap</span>
          <strong>{decision.token_budget_cap ?? '—'}</strong>
        </div>
      </div>
    </section>
  )
}

export default GateDecision

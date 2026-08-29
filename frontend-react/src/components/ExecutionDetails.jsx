import { useState } from 'react'
import AgentPipeline from './AgentPipeline'

function ExecutionDetails({ result }) {
  const [expanded, setExpanded] = useState(false)
  const decision = result.gate_decision || {}

  return (
    <div className="execution-details">
      <button className="details-toggle" type="button" onClick={() => setExpanded((open) => !open)}>
        <span>Gate Decision</span>
        <span aria-hidden="true">{expanded ? 'v' : '>'}</span>
      </button>
      {expanded ? (
        <div className="details-content">
          <div className="details-grid">
            <Detail label="Gate Type" value={decision.gate_type || '--'} />
            <Detail label="Confidence" value={`${Math.round((decision.confidence || 0) * 100)}%`} />
            <Detail label="Token Budget" value={decision.token_budget_cap ?? '--'} />
            <Detail label="Tokens Spent" value={result.tokens_spent ?? '--'} />
            <Detail label="Agents Used" value="4" />
            <Detail label="Decision" value={decision.decision || '--'} />
          </div>
          <AgentPipeline active={false} result={result} />
        </div>
      ) : null}
    </div>
  )
}

function Detail({ label, value }) {
  return <div><span>{label}</span><strong>{value}</strong></div>
}

export default ExecutionDetails
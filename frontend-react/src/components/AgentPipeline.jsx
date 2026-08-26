const pipeline = [
  { name: 'Probe Agent', status: 'Ready' },
  { name: 'Gate Layer', status: 'Ready' },
  { name: 'MAS Orchestrator', status: 'Ready' },
  { name: 'Evaluation', status: 'Ready' },
]

function AgentPipeline({ active, result }) {
  const statuses = active
    ? ['Running', 'Waiting', 'Waiting', 'Waiting']
    : result
      ? ['Complete', 'Complete', result.mas_tokens ? 'Complete' : 'Skipped', 'Complete']
      : pipeline.map((agent) => agent.status)

  return (
    <section className="pipeline-card">
      <div className="section-heading">
        <h3>Agent pipeline</h3>
      </div>

      <div className="pipeline-list">
        {pipeline.map((agent, index) => {
          const status = statuses[index]
          return (
            <div
              key={agent.name}
              className={`pipeline-item ${active && index === 0 ? 'active' : ''}`}
            >
              <span className="pipeline-index">{index + 1}</span>
              <div>
                <div className="pipeline-name">{agent.name}</div>
                <div className="pipeline-status">{status}</div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default AgentPipeline

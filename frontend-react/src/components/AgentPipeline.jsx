const pipeline = [
  { name: 'Probe Agent', status: 'Ready' },
  { name: 'Gate Layer', status: 'Ready' },
  { name: 'MAS Orchestrator', status: 'Ready' },
  { name: 'Evaluation', status: 'Ready' },
]

function AgentPipeline({ active }) {
  return (
    <section className="pipeline-card">
      <div className="section-heading">
        <h3>Agent pipeline</h3>
      </div>

      <div className="pipeline-list">
        {pipeline.map((agent, index) => {
          const isActive = active && index === 0
          return (
            <div
              key={agent.name}
              className={`pipeline-item ${isActive ? 'active' : ''}`}
            >
              <span className="pipeline-index">{index + 1}</span>
              <div>
                <div className="pipeline-name">{agent.name}</div>
                <div className="pipeline-status">{isActive ? 'Running' : agent.status}</div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

export default AgentPipeline

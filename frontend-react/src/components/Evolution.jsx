const agents = [
  {
    name: 'ReAct Agent',
    basis: 'ReAct (Yao et al., 2023)',
    role: 'Thought/action loop over the task context, then a final synthesis step.',
    route: 'Tasks with supporting context, or a high estimated depth.',
  },
  {
    name: 'Debate Agent',
    basis: 'Multiagent Debate (Du et al., 2023)',
    role: 'Proposer answers; critic reviews; proposer revises with the context and its previous solution.',
    route: 'Comparison questions ("X: A or B?") or a high estimated parallelism.',
  },
  {
    name: 'Reflexion Agent',
    basis: 'Self-Refine (Madaan et al., 2023)',
    role: 'Draft, self-critique, then refine using the context and the draft.',
    route: 'Everything else (e.g. self-contained arithmetic).',
  },
]

function Evolution() {
  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Multi-Agent System</p>
        <h2>MAS Routing</h2>
        <p>
          How an escalated task is assigned to a sub-agent. Routing uses only the task text; it never
          reads the dataset&apos;s depth or parallelism labels, which real queries do not have.
        </p>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Sub-Agent Pool</h3>
          <span className="best-indicator">Default router: text heuristic</span>
        </div>
        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head" style={{ gridTemplateColumns: '1fr 1.2fr 2fr 2fr' }}>
            <span>Agent</span>
            <span>Technique</span>
            <span>What it does</span>
            <span>Routed when</span>
          </div>
          {agents.map((agent) => (
            <div key={agent.name} className="benchmark-row" style={{ gridTemplateColumns: '1fr 1.2fr 2fr 2fr' }}>
              <strong>{agent.name}</strong>
              <span>{agent.basis}</span>
              <span>{agent.role}</span>
              <span>{agent.route}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Optional LinUCB Router</h3>
        </div>
        <p className="muted-text">
          <code>MASOrchestrator(default_strategy=&quot;bandit&quot;)</code> replaces the heuristic with a
          LinUCB contextual bandit (exploration α = 0.5 by default) over the same text-only features,
          rewarded with correctness minus 0.2 × the fraction of the budget spent. The final benchmark
          uses the heuristic router, so no bandit allocation statistics are reported.
        </p>
      </section>
    </div>
  )
}

export default Evolution

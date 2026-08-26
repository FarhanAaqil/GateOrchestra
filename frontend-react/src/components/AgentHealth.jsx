const agents = [
  ['Probe Agent', 'Healthy', '94%', '182ms', '1,248'],
  ['Gate Layer', 'Learning', '89%', '24ms', '1,248'],
  ['MAS Orchestrator', 'Degraded', '76%', '1.8s', '923'],
  ['Evaluation', 'Healthy', '98%', '31ms', '1,248'],
]

function AgentHealth() {
  return <div className="health-table"><div className="health-row health-head"><span>Agent</span><span>Health</span><span>Success Rate</span><span>Latency</span><span>Tasks</span><span>Status</span></div>{agents.map(([name, health, success, latency, tasks]) => <div className="health-row" key={name}><strong>{name}</strong><span className={`health-badge ${health.toLowerCase()}`}>{health}</span><span>{success}</span><span>{latency}</span><span>{tasks}</span><span className="health-live">● Active</span></div>)}</div>
}
export default AgentHealth
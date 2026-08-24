import { useMemo, useState } from 'react'
import './App.css'
import Navbar from './components/Navbar'
import StatCard from './components/StatCard'
import RunPanel from './components/RunPanel'
import GateDecision from './components/GateDecision'
import AgentPipeline from './components/AgentPipeline'
import ResultPanel from './components/ResultPanel'
import { runGateOrchestra } from './services/api'

function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const stats = useMemo(() => {
    const gateDecision = result?.gate_decision || {}
    const hasResult = Boolean(result)

    return [
      {
        label: 'System status',
        value: hasResult ? 'Operational' : 'Idle',
        accent: true,
        detail: hasResult ? 'Last run completed successfully' : 'Awaiting orchestration run',
      },
      {
        label: 'Token budget',
        value: hasResult ? `${result.tokens_spent ?? 0}` : '0',
        detail: 'Total tokens consumed',
      },
      {
        label: 'Tokens spent',
        value: hasResult ? `${result.tokens_spent ?? 0}` : '0',
        detail: 'Current execution total',
      },
      {
        label: 'Current gate',
        value: hasResult ? gateDecision.decision || 'N/A' : 'Waiting',
        detail: gateDecision.gate_type || 'No gate classification yet',
      },
      {
        label: 'Agents',
        value: hasResult ? '4/4' : '4',
        detail: 'Probe, gate, MAS, evaluation',
      },
    ]
  }, [result])

  const handleRun = async (payload) => {
    setLoading(true)
    setError('')

    try {
      const data = await runGateOrchestra(payload)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Could not connect to the GateOrchestra API.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <Navbar />

      <main className="dashboard-layout">
        <section className="stats-grid" aria-label="System summary">
          {stats.map((stat) => (
            <StatCard
              key={stat.label}
              label={stat.label}
              value={stat.value}
              detail={stat.detail}
              accent={stat.accent}
            />
          ))}
        </section>

        <section className="content-grid">
          <div className="left-column">
            <RunPanel onSubmit={handleRun} loading={loading} error={error} />
            <GateDecision decision={result?.gate_decision} />
          </div>

          <div className="right-column">
            <ResultPanel result={result} isLoading={loading} />
            <AgentPipeline active={loading} result={result} />
          </div>
        </section>
      </main>
    </div>
  )
}

export default App

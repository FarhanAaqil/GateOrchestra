import { useEffect, useRef, useState } from 'react'
import './App.css'
import Benchmarks from './components/Benchmarks'
import ChatComposer from './components/ChatComposer'
import ConversationView from './components/ConversationView'
import Evolution from './components/Evolution'
import Navbar from './components/Navbar'
import Schedules from './components/Schedules'
import Sidebar from './components/Sidebar'
import StatCard from './components/StatCard'
import { checkHealth, fetchBenchmarkSummary, fetchModels, runGateOrchestra } from './services/api'

// Live escalations can take minutes (rate limits, several LLM calls); give up after 5.
const RUN_TIMEOUT_MS = 5 * 60 * 1000

function createConversation(initialTitle = 'New task trace') {
  const now = new Date().toISOString()
  return { id: `trace-${Date.now()}`, title: initialTitle, messages: [], createdAt: now, updatedAt: now }
}

function mapStrategyToMethod(strategy) {
  if (!strategy) return 'GateOrchestra'
  if (strategy.includes('CoT-SC')) return 'CoT-SC'
  if (strategy.includes('Always-MAS')) return 'Always-MAS'
  if (strategy.includes('Random')) return 'RandomGate'
  if (strategy.includes('Rule-Based') || strategy.includes('RuleBased')) return 'RuleBasedGate'
  return 'GateOrchestra'
}

function App() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiOnline, setApiOnline] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [view, setView] = useState('chat')
  const [modelLabel, setModelLabel] = useState('')
  const [gateInfo, setGateInfo] = useState(undefined)
  const [runStartedAt, setRunStartedAt] = useState(null)
  const [now, setNow] = useState(() => Date.now())
  const abortRef = useRef(null)
  const [benchmark, setBenchmark] = useState(null)

  const activeConversation = conversations.find((c) => c.id === activeId)
  const latestResult = [...(activeConversation?.messages || [])].reverse().find((m) => m.result)?.result || null

  const updateConversation = (id, updater) =>
    setConversations((current) =>
      current.map((c) => (c.id === id ? updater(c) : c))
    )

  useEffect(() => {
    checkHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
    fetchModels()
      .then((data) => {
        setModelLabel(`${data.current_provider} / ${data.default_model}`)
        setGateInfo(data.trained_gate ?? null)
      })
      .catch(() => setModelLabel(''))
    fetchBenchmarkSummary()
      .then(setBenchmark)
      .catch(() => setBenchmark(null))
  }, [])

  const handleSubmit = async (
    question,
    selectedStrategy,
    taskContext = null,
    groundTruth = null,
    taskIdOverride = null,
    simulate = false,
  ) => {
    const conversation = activeConversation || createConversation()
    if (!activeConversation) {
      setConversations((current) => [conversation, ...current])
      setActiveId(conversation.id)
    }

    const taskId = taskIdOverride || `task-${Date.now()}`
    const method = mapStrategyToMethod(selectedStrategy)

    updateConversation(conversation.id, (current) => ({
      ...current,
      title: current.messages.length ? current.title : question.slice(0, 36),
      messages: [...current.messages, { id: `${taskId}-user`, role: 'user', content: question }],
      updatedAt: new Date().toISOString(),
    }))

    setLoading(true)
    setError('')
    const controller = new AbortController()
    abortRef.current = controller
    const timeout = setTimeout(() => controller.abort('timeout'), RUN_TIMEOUT_MS)
    const startedAt = Date.now()
    setRunStartedAt(startedAt)
    setNow(startedAt)

    try {
      const data = await runGateOrchestra({
        task_id: taskId,
        question,
        context: taskContext,
        ground_truth: groundTruth,
        method,
        k: 3,
        force_simulation: simulate,
      }, { signal: controller.signal })

      updateConversation(conversation.id, (current) => ({
        ...current,
        updatedAt: new Date().toISOString(),
        messages: [
          ...current.messages,
          {
            id: `${taskId}-assistant`,
            role: 'assistant',
            content: data.predicted_answer || 'No answer returned.',
            result: data,
          },
        ],
      }))
    } catch (err) {
      if (controller.signal.aborted) {
        const why = controller.signal.reason === 'timeout' ? 'Run timed out after 5 minutes' : 'Run cancelled'
        setError(
          `${why}. The server may still finish it; it will then appear under Benchmarks › Recent Executions.`,
        )
      } else {
        setError(err.message || 'Could not connect to GateOrchestra backend.')
      }
    } finally {
      clearTimeout(timeout)
      abortRef.current = null
      setRunStartedAt(null)
      setLoading(false)
    }
  }

  const handleCancel = () => abortRef.current?.abort('cancelled')

  useEffect(() => {
    if (runStartedAt === null) return undefined
    const tick = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(tick)
  }, [runStartedAt])
  const elapsedSeconds = runStartedAt === null ? 0 : Math.max(0, Math.round((now - runStartedAt) / 1000))

  const handleSelectBenchmarkTask = (task) => {
    setView('chat')
    const conversation = createConversation(`Task: ${task.task_id}`)
    setConversations((current) => [conversation, ...current])
    setActiveId(conversation.id)
    const method = gateInfo === null ? 'CoT-SC' : 'GateOrchestra'
    handleSubmit(task.question, method, task.context, task.ground_truth, task.task_id)
  }

  const handleNewChat = () => {
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setActiveId(conversation.id)
    setView('chat')
    setError('')
  }

  const openConversation = (id) => {
    setActiveId(id)
    setView('chat')
    setError('')
  }

  const deleteConversation = (id) => {
    setConversations((current) => current.filter((c) => c.id !== id))
    if (activeId === id) setActiveId(null)
  }

  // KPI cards show only values the backend actually returned.
  const decision = latestResult?.gate_decision
  const gradeLabel = !latestResult
    ? '—'
    : latestResult.is_correct === true
      ? 'Correct'
      : latestResult.is_correct === false
        ? 'Incorrect'
        : 'Ungraded'
  const gradeDetail = !latestResult
    ? 'Run a task to see its result'
    : latestResult.simulated
      ? 'Simulated run: no LLM, not graded'
      : latestResult.is_correct === null
        ? 'No ground truth supplied'
        : 'Graded against the ground truth'
  const gateRow = benchmark?.aggregated?.find((row) => row.method === 'GateOrchestra')
  const benchmarkValue = gateRow ? `${gateRow.accuracy_mean.toFixed(1)}%` : '—'
  const benchmarkDetail = gateRow
    ? `GateOrchestra accuracy, ${gateRow.n_seeds} seed(s), n=${benchmark.n_per_split ?? 'full'}`
    : 'No benchmark results recorded'

  return (
    <div className="app-shell">
      <Navbar
        apiOnline={apiOnline}
        modelLabel={modelLabel}
        onNewExecution={handleNewChat}
      />

      <div className="main-wrapper">
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          view={view}
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
          onNewChat={handleNewChat}
          onOpenConversation={openConversation}
          onDelete={deleteConversation}
          onNavigate={(nextView) => setView(nextView)}
          modelLabel={modelLabel}
          gateInfo={gateInfo}
        />

        <main className="workspace-shell">
          {/* Top KPI Stat Cards */}
          <div className="kpi-grid">
            <StatCard
              label="Gate Decision"
              value={decision?.decision || (latestResult ? 'No gate' : '—')}
              detail={
                decision
                  ? `Confidence ${Math.round(decision.confidence * 100)}% • ${decision.gate_type} gate`
                  : latestResult
                    ? `${latestResult.method} does not use a gate`
                    : 'Run a task to see the routing decision'
              }
              type="cyan"
            />
            <StatCard
              label="Tokens Spent"
              value={latestResult ? `${latestResult.tokens_spent}` : '—'}
              detail={
                latestResult
                  ? `Probe: ${latestResult.probe_tokens ?? 0} | MAS: ${latestResult.mas_tokens ?? 0}`
                  : 'Prompt + completion, as reported by the provider'
              }
              type="default"
            />
            <StatCard label="Answer" value={gradeLabel} detail={gradeDetail} type="default" />
            <StatCard
              label="Benchmark"
              value={benchmarkValue}
              detail={benchmarkDetail}
              type="default"
              badge={benchmark ? benchmark.status.replace('_', ' ').toUpperCase() : null}
            />
          </div>

          {error ? (
            <div className="api-error-banner" role="alert">
              ⚠️ {error}
            </div>
          ) : null}

          {view === 'chat' ? (
            <div className="studio-container">
              <ConversationView
                messages={activeConversation?.messages || []}
                loading={loading}
              />
              <ChatComposer
                onSubmit={handleSubmit}
                loading={loading}
                modelLabel={modelLabel}
                gateInfo={gateInfo}
                elapsedSeconds={elapsedSeconds}
                onCancel={handleCancel}
              />
            </div>
          ) : (
            <ViewComponent view={view} onSelectTask={handleSelectBenchmarkTask} benchmark={benchmark} />
          )}
        </main>
      </div>
    </div>
  )
}

function ViewComponent({ view, onSelectTask, benchmark }) {
  if (view === 'evolution') return <Evolution />
  if (view === 'benchmarks') return <Benchmarks onSelectTask={onSelectTask} benchmark={benchmark} />
  if (view === 'schedules') return <Schedules />
  return (
    <div className="welcome-box">
      <h2>{view}</h2>
      <p>Workspace section initialized.</p>
    </div>
  )
}

export default App

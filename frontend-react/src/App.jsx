import { useEffect, useState } from 'react'
import './App.css'
import Benchmarks from './components/Benchmarks'
import ChatComposer from './components/ChatComposer'
import ConversationView from './components/ConversationView'
import Evolution from './components/Evolution'
import Schedules from './components/Schedules'
import Sidebar from './components/Sidebar'
import { checkHealth, runGateOrchestra } from './services/api'

function createConversation(initialTitle = 'New conversation') {
  const now = new Date().toISOString()
  return { id: `conversation-${Date.now()}`, title: initialTitle, messages: [], createdAt: now, updatedAt: now }
}

function mapStrategyToMethod(strategy) {
  if (!strategy) return 'GateOrchestra'
  if (strategy.includes('CoT-SC')) return 'CoT-SC'
  if (strategy.includes('Always-MAS')) return 'Always-MAS'
  if (strategy.includes('Random')) return 'RandomGate'
  if (strategy.includes('Rule-Based')) return 'RuleBasedGate'
  if (strategy.includes('GateOrchestra') || strategy.includes('Auto Gate')) return 'GateOrchestra'
  return 'GateOrchestra'
}

function App() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiOnline, setApiOnline] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [view, setView] = useState('chat')

  const activeConversation = conversations.find((conversation) => conversation.id === activeId)
  const result = [...(activeConversation?.messages || [])].reverse().find((message) => message.result)?.result || null

  const updateConversation = (id, updater) =>
    setConversations((current) =>
      current.map((conversation) => (conversation.id === id ? updater(conversation) : conversation))
    )

  // Verify backend health on startup
  useEffect(() => {
    checkHealth()
      .then(() => setApiOnline(true))
      .catch(() => setApiOnline(false))
  }, [])

  const handleSubmit = async (question, selectedStrategy, taskContext = null, groundTruth = null, taskIdOverride = null) => {
    const conversation = activeConversation || createConversation()
    if (!activeConversation) {
      setConversations((current) => [conversation, ...current])
      setActiveId(conversation.id)
    }

    const taskId = taskIdOverride || `chat-${Date.now()}`
    const method = mapStrategyToMethod(selectedStrategy)

    updateConversation(conversation.id, (current) => ({
      ...current,
      title: current.messages.length ? current.title : question.slice(0, 38),
      messages: [...current.messages, { id: `${taskId}-user`, role: 'user', content: question }],
      updatedAt: new Date().toISOString(),
    }))

    setLoading(true)
    setError('')

    try {
      const data = await runGateOrchestra({
        task_id: taskId,
        question,
        context: taskContext,
        ground_truth: groundTruth,
        method,
        k: 3,
      })

      updateConversation(conversation.id, (current) => ({
        ...current,
        updatedAt: new Date().toISOString(),
        messages: [
          ...current.messages,
          {
            id: `${taskId}-assistant`,
            role: 'assistant',
            content: data.predicted_answer || 'No answer was returned.',
            result: data,
          },
        ],
      }))
    } catch (err) {
      setError(err.message || 'Could not connect to the GateOrchestra API. (Ensure `uvicorn api.main:app` is running)')
    } finally {
      setLoading(false)
    }
  }

  const handleSelectBenchmarkTask = (task) => {
    setView('chat')
    const conversation = createConversation(`Task: ${task.task_id}`)
    setConversations((current) => [conversation, ...current])
    setActiveId(conversation.id)
    handleSubmit(task.question, '⚡ GateOrchestra', task.context, task.ground_truth, task.task_id)
  }

  const handleNewChat = () => {
    const conversation = createConversation()
    setConversations((current) => [conversation, ...current])
    setActiveId(conversation.id)
    setView('chat')
    setError('')
    setSidebarOpen(false)
  }

  const openConversation = (id) => {
    setActiveId(id)
    setView('chat')
    setError('')
    setSidebarOpen(false)
  }

  const renameConversation = (id, title) =>
    updateConversation(id, (conversation) => ({ ...conversation, title, updatedAt: new Date().toISOString() }))

  const deleteConversation = (id) => {
    setConversations((current) => current.filter((conversation) => conversation.id !== id))
    if (activeId === id) setActiveId(null)
  }

  return (
    <div className="app-shell">
      <Sidebar
        result={result}
        conversations={conversations}
        activeId={activeId}
        view={view}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((collapsed) => !collapsed)}
        onNewChat={handleNewChat}
        onOpenConversation={openConversation}
        onRename={renameConversation}
        onDelete={deleteConversation}
        onNavigate={(nextView) => {
          setView(nextView)
          setSidebarOpen(false)
        }}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      {sidebarOpen ? (
        <button
          className="sidebar-scrim"
          type="button"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation"
        />
      ) : null}

      <main className="chat-shell">
        <header className="chat-header">
          <div>
            <p className="header-kicker">Workspace</p>
            <h1>GateOrchestra</h1>
          </div>
          <span className="ready-indicator">
            <span
              className="status-dot"
              style={{
                background: apiOnline === true ? '#56c982' : apiOnline === false ? '#d9534f' : '#e5bd7e',
              }}
            />
            {apiOnline === true ? 'Ready (API Online)' : apiOnline === false ? 'API Offline (Launch uvicorn)' : 'Connecting…'}
          </span>
        </header>

        {view === 'chat' ? (
          <>
            <section className="conversation-area" aria-label="Conversation">
              <ConversationView messages={activeConversation?.messages || []} loading={loading} />
              {error ? (
                <div className="api-error" role="alert">
                  {error}
                </div>
              ) : null}
            </section>
            <ChatComposer
              onSubmit={(q, s) => handleSubmit(q, s)}
              loading={loading}
              onPipeline={() => {
                if (result) {
                  setError(`Last run decision: ${result.gate_decision?.decision || 'N/A'} | Tokens: ${result.tokens_spent}`)
                } else {
                  setError('Run a task to see live pipeline execution telemetry.')
                }
              }}
            />
          </>
        ) : (
          <section className="workspace-view">
            <ViewComponent view={view} onSelectTask={handleSelectBenchmarkTask} />
          </section>
        )}
      </main>
    </div>
  )
}

function ViewComponent({ view, onSelectTask }) {
  if (view === 'evolution') return <Evolution />
  if (view === 'benchmarks') return <Benchmarks onSelectTask={onSelectTask} />
  if (view === 'schedules') return <Schedules />
  return (
    <div className="empty-view">
      <h2>{view}</h2>
      <p>This workspace is ready for a future phase.</p>
    </div>
  )
}

export default App

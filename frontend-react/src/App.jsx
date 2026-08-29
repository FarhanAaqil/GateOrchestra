import { useState } from 'react'
import './App.css'
import Benchmarks from './components/Benchmarks'
import ChatComposer from './components/ChatComposer'
import ConversationView from './components/ConversationView'
import Evolution from './components/Evolution'
import Schedules from './components/Schedules'
import Sidebar from './components/Sidebar'
import { runGateOrchestra } from './services/api'

function createConversation() {
  const now = new Date().toISOString()
  return { id: `conversation-${Date.now()}`, title: 'New conversation', messages: [], createdAt: now, updatedAt: now }
}

function App() {
  const [conversations, setConversations] = useState([])
  const [activeId, setActiveId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [view, setView] = useState('chat')
  const activeConversation = conversations.find((conversation) => conversation.id === activeId)
  const result = [...(activeConversation?.messages || [])].reverse().find((message) => message.result)?.result || null

  const updateConversation = (id, updater) => setConversations((current) => current.map((conversation) => conversation.id === id ? updater(conversation) : conversation))

  const handleSubmit = async (question) => {
    const conversation = activeConversation || createConversation()
    if (!activeConversation) {
      setActiveId(conversation.id)
    }
    const taskId = `chat-${Date.now()}`
    const userMessage = { id: `${taskId}-user`, role: 'user', content: question }
    setConversations((current) => {
      const existing = current.find((item) => item.id === conversation.id)
      const updated = {
        ...conversation,
        ...(existing || {}),
        title: existing?.messages.length ? existing.title : question.slice(0, 38),
        messages: [...(existing?.messages || []), userMessage],
        updatedAt: new Date().toISOString(),
      }
      return existing ? current.map((item) => item.id === conversation.id ? updated : item) : [updated, ...current]
    })
    setLoading(true)
    setError('')
    try {
      const data = await runGateOrchestra({
        task_id: taskId,
        question,
        context: null,
        ground_truth: null,
        method: 'RuleBasedGate',
        k: 3,
      })
      updateConversation(conversation.id, (current) => ({
        ...current,
        updatedAt: new Date().toISOString(),
        messages: [...current.messages, { id: `${taskId}-assistant`, role: 'assistant', content: data.predicted_answer || 'No answer was returned.', result: data }],
      }))
    } catch (err) {
      setError(err.message || 'Could not connect to the GateOrchestra API.')
    } finally {
      setLoading(false)
    }
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

  const renameConversation = (id, title) => updateConversation(id, (conversation) => ({ ...conversation, title, updatedAt: new Date().toISOString() }))
  const deleteConversation = (id) => {
    setConversations((current) => current.filter((conversation) => conversation.id !== id))
    if (activeId === id) setActiveId(null)
  }

  return (
    <div className="app-shell">
      <Sidebar result={result} conversations={conversations} activeId={activeId} view={view} collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((collapsed) => !collapsed)} onNewChat={handleNewChat} onOpenConversation={openConversation} onRename={renameConversation} onDelete={deleteConversation} onNavigate={(nextView) => { setView(nextView); setSidebarOpen(false) }} isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      {sidebarOpen ? <button className="sidebar-scrim" type="button" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" /> : null}
      <main className="chat-shell">
        <header className="chat-header">
          <div><p className="header-kicker">Workspace</p><h1>GateOrchestra</h1></div>
          <span className="ready-indicator"><span className="status-dot" /> Ready</span>
        </header>
        {view === 'chat' ? <>
          <section className="conversation-area" aria-label="Conversation">
            <ConversationView messages={activeConversation?.messages || []} loading={loading} />
            {error ? <div className="api-error" role="alert">{error}</div> : null}
          </section>
          <ChatComposer onSubmit={handleSubmit} loading={loading} onPipeline={() => setError('Agent Pipeline details are available after a run.')} />
        </> : <section className="workspace-view"><ViewComponent view={view} /></section>}
      </main>
    </div>
  )
}

function ViewComponent({ view }) {
  if (view === 'evolution') return <Evolution />
  if (view === 'benchmarks') return <Benchmarks />
  if (view === 'schedules') return <Schedules />
  return <div className="empty-view"><h2>{view}</h2><p>This workspace is ready for a future phase.</p></div>
}

export default App

import { ArrowUp, Check, ChevronDown, Copy, History, Loader2, Paperclip, Pencil, RefreshCw, Square, ThumbsDown, ThumbsUp, Wrench, X } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import ChatSidebar from '../components/ChatSidebar'
import KeyDialog from '../components/KeyDialog'
import { Markdown } from '../components/Markdown'
import { GATE_LABELS, chatApi, streamChat, type ChatEvent, type ChatMessage, type Conversation, type GateScore, type KeyAttempt, type ToolCall, type TurnMeta, type UploadedDocument } from '../lib/chat'
import { useApp } from '../lib/context'
import { describeRoute, formatSeconds } from '../lib/route'

const GATE_MODE_KEY = 'gateorchestra.gateMode'
const GATE_MODES = [
  { id: 'auto', label: 'Auto (recommended)' },
  { id: 'learned', label: 'Learned gate' },
  { id: 'rule_based', label: 'Rule-based gate' },
  { id: 'always_stop', label: 'Always answer directly' },
  { id: 'always_escalate', label: 'Always research with tools' },
]
const SUGGESTIONS = [
  'Explain how a hash map works, with a Python example',
  'If I invest $5,000 at 7% a year, how much do I have after 12 years?',
  'Who won the 2022 FIFA World Cup and who was the top scorer?',
  'Write a SQL query for the top 3 customers by total order value',
]

interface Live {
  stage: 'probe' | 'gate' | 'tools' | 'writing'
  probe?: Extract<ChatEvent, { type: 'probe' }>
  gate?: Extract<ChatEvent, { type: 'gate' }>
  tools: ToolCall[]
  text: string
}

function readMode(): string {
  try {
    return localStorage.getItem(GATE_MODE_KEY) || 'auto'
  } catch {
    return 'auto'
  }
}

function RouteBadges({ meta }: { meta: Partial<TurnMeta> }) {
  if (meta.stopped) return <p className="mt-2 text-xs text-wine">Stopped</p>
  const route = describeRoute(meta)
  if (!route) return null
  return (
    <div className="mt-2 min-w-0 text-xs text-forest/70">
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          className={`rounded-md px-2.5 py-0.5 font-medium ${route.escalated ? 'border border-clay/50 bg-clay/10 text-ink' : meta.cached ? 'border border-forest/15 bg-paper text-forest' : 'border border-forest/15 bg-sage/25 text-ink'}`}
          title={route.explanation}
        >
          {route.headline}
        </span>
        {meta.total_tokens != null ? <span className="rounded-md border border-forest/15 px-2 py-0.5">{meta.total_tokens.toLocaleString()} tokens</span> : null}
        {meta.latency_ms != null ? <span className="rounded-md border border-forest/15 px-2 py-0.5">{formatSeconds(meta.latency_ms)}</span> : null}
      </div>
      <details className="group mt-1">
        <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-forest/70 hover:text-forest">
          How this was answered
          <ChevronDown className="h-3 w-3 transition-transform group-open:rotate-180" />
        </summary>
        <p className="mt-1 text-forest/70">{route.explanation}</p>
        <dl className="mt-1.5 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
          {route.details.map((d) => (
            <div key={d.label} className="contents">
              <dt className="text-forest/70">{d.label}</dt>
              <dd className="break-words text-forest">{d.value}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  )
}

function ToolList({ tools }: { tools: ToolCall[] }) {
  if (!tools.length) return null
  return (
    <details className="group mt-2 rounded-lg border border-forest/10 bg-paper/60 px-3 py-2 text-sm">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-forest">
        <Wrench className="h-4 w-4 text-wine" /> {tools.length} tool call{tools.length === 1 ? '' : 's'}
        <ChevronDown className="ml-auto h-4 w-4 transition-transform group-open:rotate-180" />
      </summary>
      <ol className="mt-2 space-y-2">
        {tools.map((t, i) => (
          <li key={i} className="rounded-xl bg-canvas p-2">
            <p className="font-mono text-xs font-semibold text-ink">{t.name}({JSON.stringify(t.input)})</p>
            <pre tabIndex={0} className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap font-mono text-xs text-forest/80">{t.output}</pre>
          </li>
        ))}
      </ol>
    </details>
  )
}

/** A question you asked, which you can edit and send again (later messages are replaced). */
function UserBubble({ message, busy, onEdit }: { message: ChatMessage; busy: boolean; onEdit?: (text: string) => void }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(message.content)
  const save = () => {
    const text = draft.trim()
    if (!text || !onEdit) return
    setEditing(false)
    onEdit(text)
  }
  return (
    <div className="group flex flex-col items-end gap-1.5">
      {message.meta?.documents?.length ? (
        <ul className="flex max-w-[85%] flex-wrap justify-end gap-1.5" aria-label="Attached documents">
          {message.meta.documents.map((name) => (
            <li key={name} className="inline-flex items-center gap-1 rounded-md border border-forest/15 bg-canvas px-2.5 py-1 text-xs text-forest">
              <Paperclip className="h-3 w-3 text-forest/70" /> {name}
            </li>
          ))}
        </ul>
      ) : null}
      {editing ? (
        <div className="w-full max-w-[85%] rounded-xl border border-forest/20 bg-paper p-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                save()
              }
              if (e.key === 'Escape') setEditing(false)
            }}
            rows={Math.min(6, Math.max(2, draft.split('\n').length))}
            aria-label="Edit your question"
            className="w-full resize-none bg-transparent px-2 py-1 text-sm text-ink outline-none"
            autoFocus
          />
          <div className="flex justify-end gap-2">
            <button onClick={() => setEditing(false)} className="btn btn-ghost py-1!">Cancel</button>
            <button onClick={save} disabled={!draft.trim() || busy} className="btn btn-primary py-1!">Send</button>
          </div>
          <p className="px-2 pt-1 text-[11px] text-forest/70">Sending replaces this question and everything after it.</p>
        </div>
      ) : (
        <div className="flex max-w-[85%] items-start gap-1">
          {onEdit ? (
            <button
              onClick={() => {
                setDraft(message.content)
                setEditing(true)
              }}
              disabled={busy}
              className="mt-1.5 rounded-lg p-1.5 text-forest/70 transition-opacity hover:bg-forest/5 hover:text-ink focus-visible:opacity-100 disabled:opacity-30 sm:opacity-0 sm:group-hover:opacity-100"
              aria-label="Edit this question"
              title="Edit and send again"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
          ) : null}
          <p className="whitespace-pre-wrap rounded-xl bg-forest px-4 py-2.5 text-sm text-paper">{message.content}</p>
        </div>
      )}
    </div>
  )
}

function AssistantBubble({ message, onRate, onRegenerate }: { message: ChatMessage; onRate: (id: string, rating: -1 | 0 | 1) => void; onRegenerate?: () => void }) {
  const [copied, setCopied] = useState(false)
  const copy = () => {
    navigator.clipboard?.writeText(message.content).then(() => {
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    }).catch(() => undefined)
  }
  const rateable = !message.id.startsWith('stopped-') && !message.meta?.stopped
  const rating = message.rating ?? 0
  return (
    <div className="group">
      <Markdown text={message.content} />
      {message.meta?.tools?.length ? <ToolList tools={message.meta.tools} /> : null}
      <div className="flex items-start justify-between gap-2">
        <RouteBadges meta={message.meta ?? {}} />
        <div className="flex shrink-0 items-center">
          {rateable ? (
            <>
              <button
                onClick={() => onRate(message.id, rating === 1 ? 0 : 1)}
                className={`rounded-lg p-2 hover:bg-forest/5 ${rating === 1 ? 'text-forest' : 'text-forest/70'}`}
                aria-label="Helpful"
                aria-pressed={rating === 1}
              >
                <ThumbsUp className="h-4 w-4" fill={rating === 1 ? 'currentColor' : 'none'} />
              </button>
              <button
                onClick={() => onRate(message.id, rating === -1 ? 0 : -1)}
                className={`rounded-lg p-2 hover:bg-forest/5 ${rating === -1 ? 'text-wine' : 'text-forest/70'}`}
                aria-label="Not helpful"
                aria-pressed={rating === -1}
              >
                <ThumbsDown className="h-4 w-4" fill={rating === -1 ? 'currentColor' : 'none'} />
              </button>
            </>
          ) : null}
          {onRegenerate && rateable ? (
            <button onClick={onRegenerate} className="rounded-lg p-2 text-forest/70 hover:bg-forest/5 hover:text-forest" aria-label="Regenerate answer" title="Try again">
              <RefreshCw className="h-4 w-4" />
            </button>
          ) : null}
          <button onClick={copy} className="rounded-lg p-2 text-forest/70 transition-opacity hover:bg-forest/5 hover:text-forest focus-visible:opacity-100 sm:opacity-0 sm:group-hover:opacity-100" aria-label={copied ? 'Copied' : 'Copy answer'} title={copied ? 'Copied' : 'Copy answer'}>
            {copied ? <Check className="h-4 w-4 text-forest" /> : <Copy className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </div>
  )
}

function RoutingPanel({ scores, margin }: { scores: GateScore[]; margin: number }) {
  if (!scores.length) return null
  return (
    <details className="group border-b border-forest/10 bg-paper/50 px-4 py-2 text-xs">
      <summary className="flex cursor-pointer list-none items-center gap-2 text-forest">
        How auto routing chooses
        <span className="rounded-md bg-forest px-2 py-0.5 font-mono text-[10px] text-sage">
          now: {GATE_LABELS[scores.find((s) => s.chosen)?.gate ?? 'rule_based']}
        </span>
        <ChevronDown className="ml-auto h-3.5 w-3.5 transition-transform group-open:rotate-180" />
      </summary>
      <p className="mt-2 text-forest/70">
        The cheapest gate whose rated accuracy is within {Math.round(margin * 100)} points of the best. Ratings on answers teach it.
      </p>
      <table className="mt-2 w-full font-mono text-[11px]">
        <thead className="text-forest/70">
          <tr><th className="text-left font-normal">Gate</th><th className="text-right font-normal">≈ tokens</th><th className="text-right font-normal">escalates</th><th className="text-right font-normal">accuracy</th><th className="text-right font-normal">ratings</th></tr>
        </thead>
        <tbody>
          {scores.map((s) => (
            <tr key={s.gate} className={s.chosen ? 'font-semibold text-ink' : s.eligible ? 'text-forest' : 'text-forest/70'}>
              <td>{GATE_LABELS[s.gate] ?? s.gate}{s.chosen ? ' ←' : ''}</td>
              <td className="text-right">{Math.round(s.expected_tokens).toLocaleString()}</td>
              <td className="text-right">{Math.round(s.escalate_rate * 100)}%</td>
              <td className="text-right">{Math.round(s.accuracy * 100)}%</td>
              <td className="text-right">{s.up} / {s.down}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  )
}

function useElapsedSeconds(): number {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const started = Date.now()
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => window.clearInterval(timer)
  }, [])
  return seconds
}

function LiveBubble({ live }: { live: Live }) {
  const elapsed = useElapsedSeconds()
  const steps = [
    { id: 'probe', label: live.probe ? `Quick check: ${live.probe.task_type}, ${Math.round(live.probe.consistency * 100)}% agree` : 'Quick check…' },
    { id: 'gate', label: live.gate ? `${GATE_LABELS[live.gate.gate] ?? live.gate.gate}: ${live.gate.decision === 'ESCALATE' ? 'research' : 'answer directly'}` : 'Deciding' },
    ...(live.gate?.decision === 'ESCALATE' || live.tools.length ? [{ id: 'tools', label: live.tools.length ? `${live.tools.length} tool call(s)` : 'Researching…' }] : []),
    { id: 'writing', label: 'Writing answer' },
  ]
  const order = ['probe', 'gate', 'tools', 'writing']
  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-1.5" aria-live="polite">
        {steps.map((s) => {
          const done = order.indexOf(s.id) < order.indexOf(live.stage) || (s.id === 'writing' && live.text)
          const active = s.id === live.stage
          return (
            <span key={s.id} className={`rounded-md px-2.5 py-0.5 font-mono text-[11px] ${active ? 'border border-forest bg-forest text-paper' : done ? 'border border-forest/20 bg-paper text-forest' : 'border border-forest/10 text-forest/70'}`}>
              {s.label}
            </span>
          )
        })}
      </div>
      {live.text ? (
        <Markdown text={live.text + ' ▍'} />
      ) : (
        <div className="flex items-center gap-3">
          <div className="h-5 w-16 animate-pulse rounded bg-forest/10 motion-reduce:animate-none" />
          <span className="text-xs text-forest/70" role="status">
            {elapsed < 8 ? `Working… ${elapsed} s` : `Still working… ${elapsed} s. Harder questions take longer.`}
          </span>
        </div>
      )}
      <ToolList tools={live.tools} />
    </div>
  )
}

export default function Chat() {
  const { backend, isAdmin } = useApp()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [live, setLive] = useState<Live | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [keyPrompt, setKeyPrompt] = useState<{ attempts: KeyAttempt[]; message: string } | null>(null)
  const [gateMode, setGateModeState] = useState(readMode)
  const [routing, setRouting] = useState<{ margin: number; gates: GateScore[] }>({ margin: 0.02, gates: [] })
  const [documents, setDocuments] = useState<UploadedDocument[]>([])
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)
  const [historyOpen, setHistoryOpen] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const textRef = useRef('')
  const localId = useRef(0)

  const setGateMode = (mode: string) => {
    setGateModeState(mode)
    try {
      localStorage.setItem(GATE_MODE_KEY, mode)
    } catch {
      // Storage may be unavailable; the choice still applies to this session.
    }
  }

  const loadConversations = useCallback(() => {
    chatApi.conversations().then((d) => setConversations(d.conversations)).catch(() => undefined)
  }, [])

  const loadRouting = useCallback(() => {
    chatApi.gates().then((d) => setRouting({ margin: d.margin, gates: d.gates })).catch(() => undefined)
  }, [])

  const loadDocuments = useCallback(() => {
    chatApi.documents().then((d) => setDocuments(d.documents)).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (backend === 'online') {
      loadConversations()
      loadRouting()
      loadDocuments()
    }
  }, [backend, loadConversations, loadRouting, loadDocuments])

  const upload = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true)
    setError(null)
    try {
      for (const file of Array.from(files)) {
        const doc = await chatApi.uploadDocument(file)
        setSelectedDocs((ids) => [...ids, doc.id])
      }
      loadDocuments()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setUploading(false)
      if (fileRef.current) fileRef.current.value = ''
    }
  }

  const toggleDoc = (id: string) =>
    setSelectedDocs((ids) => (ids.includes(id) ? ids.filter((x) => x !== id) : [...ids, id]))

  const removeDoc = async (id: string) => {
    await chatApi.removeDocument(id).catch(() => undefined)
    setSelectedDocs((ids) => ids.filter((x) => x !== id))
    loadDocuments()
  }

  const rate = async (id: string, rating: -1 | 0 | 1) => {
    setMessages((m) => m.map((x) => (x.id === id ? { ...x, rating } : x)))
    try {
      await chatApi.rate(id, rating)
      loadRouting()
    } catch (err) {
      setError((err as Error).message)
    }
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, live?.text, live?.stage])

  const openConversation = async (id: string) => {
    if (live) return
    setHistoryOpen(false)
    setActiveId(id)
    setError(null)
    try {
      setMessages((await chatApi.conversation(id)).messages)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  const newChat = () => {
    if (live) return
    setHistoryOpen(false)
    setActiveId(null)
    setMessages([])
    setError(null)
    inputRef.current?.focus()
  }

  const renameConversation = async (id: string, title: string) => {
    await chatApi.rename(id, title).catch((err: Error) => setError(err.message))
    loadConversations()
  }

  const deleteConversation = async (id: string) => {
    await chatApi.remove(id).catch(() => undefined)
    if (id === activeId) newChat()
    loadConversations()
  }

  const send = async (text: string, options: { replaceLast?: boolean; editFrom?: string } = {}) => {
    const attachedNames = documents.filter((d) => selectedDocs.includes(d.id)).map((d) => d.name)
    // A file on its own is a valid message: ask about it.
    const message =
      text.trim() ||
      (attachedNames.length ? `Summarise ${attachedNames.length === 1 ? 'the attached document' : 'the attached documents'}.` : '')
    if (!message || live) return
    if (options.editFrom) setMessages((m) => m.slice(0, Math.max(0, m.findIndex((x) => x.id === options.editFrom))))
    else if (options.replaceLast) setMessages((m) => m.slice(0, Math.max(0, m.length - 2)))
    localId.current += 1
    const optimistic: ChatMessage = {
      id: `local-${localId.current}`,
      role: 'user',
      content: message,
      meta: attachedNames.length ? { documents: attachedNames } : undefined,
    }
    setMessages((m) => [...m, optimistic])
    setInput('')
    setError(null)
    setLive({ stage: 'probe', tools: [], text: '' })
    textRef.current = ''
    let savedId = ''
    const controller = new AbortController()
    abortRef.current = controller
    try {
      await streamChat(
        { message, conversation_id: activeId, gate_mode: gateMode, document_ids: selectedDocs.length ? selectedDocs : null, replace_last: !!options.replaceLast, edit_from: options.editFrom ?? null },
        (event) => {
          if (event.type === 'meta') {
            setActiveId(event.conversation_id)
            // The saved question's real id, so it can be edited later in this session.
            savedId = event.user_message_id
            setMessages((m) => m.map((x) => (x.id === optimistic.id ? { ...x, id: event.user_message_id } : x)))
          } else if (event.type === 'probe') {
            setLive((l) => l && { ...l, probe: event, stage: 'gate' })
          } else if (event.type === 'gate') {
            setLive((l) => l && { ...l, gate: event, stage: event.decision === 'ESCALATE' ? 'tools' : 'writing' })
          } else if (event.type === 'tool') {
            setLive((l) => l && { ...l, tools: [...l.tools, event] })
          } else if (event.type === 'answer_start') {
            setLive((l) => l && { ...l, stage: 'writing' })
          } else if (event.type === 'delta') {
            textRef.current += event.text
            setLive((l) => l && { ...l, text: textRef.current })
          } else if (event.type === 'done') {
            const content = textRef.current
            setMessages((m) => [...m, { id: event.message_id, role: 'assistant', content, meta: event.meta }])
            setLive(null)
          } else if (event.type === 'error') {
            setMessages((m) => m.filter((x) => x.id !== optimistic.id && x.id !== savedId))
            // Only the operator can add a server key; everyone else just hears that it is busy.
            if (event.code === 'needs_key' && isAdmin) setKeyPrompt({ attempts: event.attempts ?? [], message })
            else if (event.code === 'needs_key') setError('The assistant is at capacity right now. Try again in a minute.')
            else setError(event.message)
          }
        },
        controller.signal,
      )
    } catch (err) {
      if (controller.signal.aborted) {
        const partial = textRef.current
        if (partial.trim()) {
          localId.current += 1
          setMessages((m) => [...m, { id: `stopped-${localId.current}`, role: 'assistant', content: partial, meta: { stopped: true } }])
        }
        setError(partial.trim() ? 'Stopped. The partial answer was kept.' : 'Stopped.')
      } else {
        setError((err as Error).message)
        setMessages((m) => m.filter((x) => x.id !== optimistic.id && x.id !== savedId))
      }
    } finally {
      abortRef.current = null
      setLive(null)
      loadConversations()
      loadRouting()
    }
  }

  const lastUser = [...messages].reverse().find((m) => m.role === 'user')
  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id
  const regenerate = () => {
    if (lastUser && !live) send(lastUser.content, { replaceLast: true })
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send(input)
    }
  }

  return (
    <div className="mx-auto grid h-[calc(100vh-7.5rem)] w-full max-w-7xl gap-4 lg:grid-cols-[250px_minmax(0,1fr)]">
      <aside className="card hidden min-h-0 flex-col p-3 lg:flex" aria-label="Conversation history">
        <ChatSidebar
          conversations={conversations}
          activeId={activeId}
          busy={!!live}
          documents={documents}
          selectedDocs={selectedDocs}
          onOpen={openConversation}
          onNew={newChat}
          onDelete={deleteConversation}
          onRename={renameConversation}
          onToggleDoc={toggleDoc}
          onRemoveDoc={removeDoc}
        />
      </aside>

      {historyOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Conversation history">
          <div className="absolute inset-0 bg-forest/50" onClick={() => setHistoryOpen(false)} />
          <div className="absolute inset-y-0 left-0 flex w-[85%] max-w-sm flex-col bg-canvas p-3 shadow-xl">
            <button onClick={() => setHistoryOpen(false)} className="mb-2 self-end rounded-lg p-2 text-forest hover:bg-forest/5" aria-label="Close history">
              <X className="h-4 w-4" />
            </button>
            <div className="min-h-0 flex-1">
              <ChatSidebar
                conversations={conversations}
          activeId={activeId}
          busy={!!live}
          documents={documents}
          selectedDocs={selectedDocs}
          onOpen={openConversation}
          onNew={newChat}
          onDelete={deleteConversation}
          onRename={renameConversation}
          onToggleDoc={toggleDoc}
          onRemoveDoc={removeDoc}
              />
            </div>
          </div>
        </div>
      ) : null}

      <section className="card flex min-h-0 flex-col p-0" aria-label="Chat">
        <div className="flex flex-wrap items-center gap-2 border-b border-forest/10 px-4 py-2.5">
          <h1 className="text-lg font-semibold text-ink">Chat</h1>
          <button onClick={() => setHistoryOpen(true)} className="inline-flex items-center gap-1.5 rounded-md border border-forest/15 px-3 py-1 text-xs text-forest hover:bg-forest/5 lg:hidden">
            <History className="h-3.5 w-3.5" /> History
          </button>
          <button onClick={newChat} disabled={!!live} className="rounded-md border border-forest/15 px-3 py-1 text-xs text-forest hover:bg-forest/5 disabled:opacity-50 lg:hidden">New chat</button>
          <label className="ml-auto flex items-center gap-2 text-xs text-forest/70">
            Mode
            <select value={gateMode} onChange={(e) => setGateMode(e.target.value)} disabled={!!live} className="max-w-[11rem] rounded-md border border-forest/15 bg-canvas px-2.5 py-1 text-xs text-ink disabled:opacity-60 sm:max-w-none" title="How the assistant decides how much effort a question needs">
              {GATE_MODES.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>
          </label>
        </div>

        {gateMode === 'auto' ? <RoutingPanel scores={routing.gates} margin={routing.margin} /> : null}

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8" tabIndex={0} role="log" aria-label="Messages">
          {!messages.length && !live ? (
            <div className="mx-auto mt-6 max-w-2xl text-center">
              <h2 className="text-xl font-semibold text-ink">Ask anything</h2>
              <p className="mt-2 text-sm text-forest/70">
                Easy questions get a direct answer. Harder ones are researched with tools first, so you get accuracy without paying for effort you do not need.
              </p>
              <ul className="mt-3 flex flex-wrap justify-center gap-1.5 text-xs text-forest/70" aria-label="What it can do">
                {['Attach files and ask about them', 'Looks facts up on Wikipedia', 'Runs Python safely', 'Learns from your ratings'].map((t) => (
                  <li key={t} className="rounded-md border border-forest/15 px-2.5 py-0.5">{t}</li>
                ))}
              </ul>
              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => send(s)} disabled={backend !== 'online'} className="rounded-lg border border-forest/15 bg-canvas px-4 py-3 text-left text-sm text-forest hover:bg-forest/5 disabled:opacity-50">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : null}
          <div className="mx-auto max-w-3xl space-y-6">
            {messages.map((m) =>
              m.role === 'user' ? (
                <UserBubble
                  key={m.id}
                  message={m}
                  busy={!!live}
                  onEdit={m.id.startsWith('local-') ? undefined : (text) => send(text, { editFrom: m.id })}
                />
              ) : (
                <AssistantBubble key={m.id} message={m} onRate={rate} onRegenerate={m.id === lastAssistantId ? () => regenerate() : undefined} />
              ),
            )}
            {live ? <LiveBubble live={live} /> : null}
            {error ? <p className="rounded-lg bg-wine/10 px-4 py-2.5 text-sm text-wine" role="alert">{error}</p> : null}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="border-t border-forest/10 p-3 sm:px-8">
          {selectedDocs.length ? (
            <p className="mx-auto mb-1.5 max-w-3xl px-2 text-xs text-forest/70">
              Answering from {selectedDocs.length} document{selectedDocs.length === 1 ? '' : 's'}:{' '}
              {documents.filter((d) => selectedDocs.includes(d.id)).map((d) => d.name).join(', ')}
            </p>
          ) : null}
          <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-forest/15 bg-canvas px-3 py-2 focus-within:ring-2 focus-within:ring-forest/30">
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".txt,.md,.markdown,.csv,.json,.pdf,.py,.js,.ts,.tsx,.html,.css,.sql,.yaml,.yml,.xml,.java,.c,.cpp,.go,.rs,.rb"
              className="hidden"
              onChange={(e) => upload(e.target.files)}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={uploading || backend !== 'online'}
              className="rounded-lg p-2 text-forest/70 hover:bg-forest/5 hover:text-forest disabled:opacity-40"
              aria-label="Attach a document"
              title="Attach a text, code or PDF file"
            >
              {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Paperclip className="h-4 w-4" />}
            </button>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              rows={Math.min(6, Math.max(1, input.split('\n').length))}
              placeholder={backend === 'online' ? 'Ask a question…' : 'Connecting to the API…'}
              aria-label="Message"
              className="max-h-40 min-h-[2.25rem] flex-1 resize-none bg-transparent py-1.5 text-sm text-ink outline-none"
            />
            {live ? (
              <button onClick={() => abortRef.current?.abort()} className="rounded-lg bg-wine p-2 text-canvas" aria-label="Stop">
                <Square className="h-4 w-4" />
              </button>
            ) : (
              <button onClick={() => send(input)} disabled={(!input.trim() && !selectedDocs.length) || backend !== 'online'} className="rounded-lg bg-forest p-2 text-paper disabled:opacity-30" aria-label="Send">
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>
          <p className="mx-auto mt-1.5 hidden max-w-3xl px-3 text-[11px] text-forest/70 sm:block">
            Enter to send · Shift+Enter for a new line · Answers can be wrong, so check anything important.
          </p>
        </div>
      </section>

      {keyPrompt ? (
        <KeyDialog
          attempts={keyPrompt.attempts}
          onClose={() => setKeyPrompt(null)}
          onSaved={() => {
            const retry = keyPrompt.message
            setKeyPrompt(null)
            send(retry)
          }}
        />
      ) : null}
    </div>
  )
}

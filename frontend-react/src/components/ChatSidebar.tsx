import { Check, FileText, MessageSquarePlus, Pencil, Search, Trash2, X } from 'lucide-react'
import { useMemo, useState, type KeyboardEvent } from 'react'
import type { Conversation, UploadedDocument } from '../lib/chat'
import { filterConversations, groupConversations } from '../lib/conversations'

interface Props {
  conversations: Conversation[]
  activeId: string | null
  busy: boolean
  documents: UploadedDocument[]
  selectedDocs: string[]
  onOpen: (id: string) => void
  onNew: () => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  onToggleDoc: (id: string) => void
  onRemoveDoc: (id: string) => void
}

/** Conversation history and documents. Used in the desktop sidebar and the phone drawer. */
export default function ChatSidebar({
  conversations,
  activeId,
  busy,
  documents,
  selectedDocs,
  onOpen,
  onNew,
  onDelete,
  onRename,
  onToggleDoc,
  onRemoveDoc,
}: Props) {
  const [query, setQuery] = useState('')
  const [editingId, setEditingId] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const groups = useMemo(() => groupConversations(filterConversations(conversations, query)), [conversations, query])

  const startEdit = (c: Conversation) => {
    setEditingId(c.id)
    setDraft(c.title)
  }
  const commit = () => {
    const title = draft.trim()
    if (editingId && title) onRename(editingId, title)
    setEditingId(null)
  }
  const onEditKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') commit()
    if (e.key === 'Escape') setEditingId(null)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <button
        onClick={onNew}
        disabled={busy}
        className="mb-3 inline-flex items-center justify-center gap-2 rounded-full bg-ink px-4 py-2 text-sm font-semibold text-soft disabled:opacity-50"
      >
        <MessageSquarePlus className="h-4 w-4" /> New chat
      </button>

      {conversations.length > 5 ? (
        <label className="relative mb-2 block">
          <span className="sr-only">Search conversations</span>
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-navy/40" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search chats"
            className="w-full rounded-full border border-sky bg-soft py-1.5 pl-8 pr-3 text-xs text-ink outline-none focus:ring-2 focus:ring-navy/30"
          />
        </label>
      ) : null}

      <div className="min-h-0 flex-1 overflow-y-auto" aria-label="Conversations">
        {groups.map((group) => (
          <section key={group.label} className="mb-2">
            <h3 className="px-3 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wide text-navy/50">{group.label}</h3>
            <ul className="space-y-0.5">
              {group.items.map((c) => (
                <li key={c.id} className="group flex items-center">
                  {editingId === c.id ? (
                    <div className="flex min-w-0 flex-1 items-center gap-1 px-1">
                      <input
                        autoFocus
                        value={draft}
                        onChange={(e) => setDraft(e.target.value)}
                        onKeyDown={onEditKey}
                        maxLength={80}
                        aria-label="Conversation title"
                        className="min-w-0 flex-1 rounded-lg border border-sky bg-soft px-2 py-1.5 text-sm text-ink outline-none focus:ring-2 focus:ring-navy/30"
                      />
                      <button onClick={commit} className="rounded-lg p-1.5 text-navy hover:bg-ice" aria-label="Save title">
                        <Check className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <button
                        onClick={() => onOpen(c.id)}
                        disabled={busy}
                        aria-current={c.id === activeId ? 'true' : undefined}
                        className={`min-w-0 flex-1 truncate rounded-xl px-3 py-2 text-left text-sm disabled:opacity-60 ${
                          c.id === activeId ? 'bg-navy text-ice' : 'text-navy hover:bg-ice'
                        }`}
                        title={c.title}
                      >
                        {c.title}
                      </button>
                      <button
                        onClick={() => startEdit(c)}
                        className="ml-0.5 rounded-lg p-1.5 text-navy/50 hover:bg-ice hover:text-navy focus-visible:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
                        aria-label={`Rename ${c.title}`}
                      >
                        <Pencil className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => onDelete(c.id)}
                        className="rounded-lg p-1.5 text-navy/50 hover:bg-wine/10 hover:text-wine focus-visible:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
                        aria-label={`Delete ${c.title}`}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  )}
                </li>
              ))}
            </ul>
          </section>
        ))}
        {!conversations.length ? <p className="px-3 py-2 text-xs text-navy/50">No conversations yet. Ask something to start one.</p> : null}
        {conversations.length && !groups.length ? <p className="px-3 py-2 text-xs text-navy/50">No chats match “{query}”.</p> : null}
      </div>

      <div className="mt-3 border-t border-sky/60 pt-3">
        <p className="mb-1.5 flex items-center justify-between px-1 text-xs font-semibold text-navy">
          Documents
          <span className="font-normal text-navy/60">
            {selectedDocs.length}/{documents.length} in use
          </span>
        </p>
        <ul className="max-h-40 space-y-1 overflow-y-auto" aria-label="Documents">
          {documents.map((d) => (
            <li key={d.id} className="group flex items-center gap-1.5 rounded-lg px-1 py-1 text-xs text-navy hover:bg-ice">
              <input
                type="checkbox"
                checked={selectedDocs.includes(d.id)}
                onChange={() => onToggleDoc(d.id)}
                aria-label={`Use ${d.name}`}
                className="accent-[#1B263B]"
              />
              <FileText className="h-3.5 w-3.5 shrink-0 text-wine" />
              <span className="min-w-0 flex-1 truncate" title={`${d.name} · ${d.chunks} passages`}>
                {d.name}
              </span>
              <button
                onClick={() => onRemoveDoc(d.id)}
                className="rounded p-1 text-navy/50 hover:text-wine focus-visible:opacity-100 sm:opacity-0 sm:group-hover:opacity-100"
                aria-label={`Remove ${d.name}`}
              >
                <X className="h-3 w-3" />
              </button>
            </li>
          ))}
          {!documents.length ? <li className="px-1 text-xs text-navy/60">Attach a file with the paperclip to ask about it.</li> : null}
        </ul>
      </div>
    </div>
  )
}

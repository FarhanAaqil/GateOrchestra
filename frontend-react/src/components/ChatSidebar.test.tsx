import { fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import type { Conversation } from '../lib/chat'
import ChatSidebar from './ChatSidebar'

const NOW = Date.now() / 1000

function convo(id: string, title: string, secondsAgo = 60): Conversation {
  return { id, title, updated_at: NOW - secondsAgo }
}

function setup(overrides: Partial<React.ComponentProps<typeof ChatSidebar>> = {}) {
  const props = {
    conversations: [convo('a', 'Nile versus Amazon'), convo('b', 'Python hash map', 90_000)],
    activeId: 'a',
    busy: false,
    documents: [{ id: 'd1', name: 'notes.md', chunks: 4, chars: 1200 }],
    selectedDocs: ['d1'],
    onOpen: vi.fn(),
    onNew: vi.fn(),
    onDelete: vi.fn(),
    onRename: vi.fn(),
    onToggleDoc: vi.fn(),
    onRemoveDoc: vi.fn(),
    ...overrides,
  }
  render(<ChatSidebar {...props} />)
  return props
}

it('groups chats by recency and marks the open one', () => {
  setup()
  expect(screen.getByRole('heading', { name: 'Today' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Yesterday' })).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Nile versus Amazon' })).toHaveAttribute('aria-current', 'true')
})

it('opens a chat and starts a new one', () => {
  const props = setup()
  fireEvent.click(screen.getByRole('button', { name: 'Python hash map' }))
  expect(props.onOpen).toHaveBeenCalledWith('b')
  fireEvent.click(screen.getByRole('button', { name: /new chat/i }))
  expect(props.onNew).toHaveBeenCalled()
})

it('renames a chat with Enter and cancels with Escape', () => {
  const props = setup()
  fireEvent.click(screen.getByRole('button', { name: 'Rename Python hash map' }))
  const input = screen.getByLabelText('Conversation title')
  fireEvent.change(input, { target: { value: 'Hash maps in Python' } })
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(props.onRename).toHaveBeenCalledWith('b', 'Hash maps in Python')

  fireEvent.click(screen.getByRole('button', { name: 'Rename Nile versus Amazon' }))
  fireEvent.keyDown(screen.getByLabelText('Conversation title'), { key: 'Escape' })
  expect(props.onRename).toHaveBeenCalledTimes(1)
  expect(screen.queryByLabelText('Conversation title')).not.toBeInTheDocument()
})

it('does not save an empty title', () => {
  const props = setup()
  fireEvent.click(screen.getByRole('button', { name: 'Rename Nile versus Amazon' }))
  const input = screen.getByLabelText('Conversation title')
  fireEvent.change(input, { target: { value: '   ' } })
  fireEvent.keyDown(input, { key: 'Enter' })
  expect(props.onRename).not.toHaveBeenCalled()
})

it('deletes a chat', () => {
  const props = setup()
  fireEvent.click(screen.getByRole('button', { name: 'Delete Python hash map' }))
  expect(props.onDelete).toHaveBeenCalledWith('b')
})

it('searches once there are enough chats, and says when nothing matches', () => {
  const many = Array.from({ length: 7 }, (_, i) => convo(`c${i}`, i === 3 ? 'Tokyo flight distance' : `Chat number ${i}`))
  setup({ conversations: many, activeId: null })
  fireEvent.change(screen.getByPlaceholderText('Search chats'), { target: { value: 'tokyo' } })
  expect(screen.getByRole('button', { name: 'Tokyo flight distance' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Chat number 1' })).not.toBeInTheDocument()
  fireEvent.change(screen.getByPlaceholderText('Search chats'), { target: { value: 'zzz' } })
  expect(screen.getByText(/No chats match/)).toBeInTheDocument()
})

it('hides search for a short list and explains an empty one', () => {
  setup({ conversations: [], activeId: null })
  expect(screen.queryByPlaceholderText('Search chats')).not.toBeInTheDocument()
  expect(screen.getByText(/No conversations yet/)).toBeInTheDocument()
})

it('blocks switching or starting chats while an answer is being written', () => {
  const props = setup({ busy: true })
  expect(screen.getByRole('button', { name: /new chat/i })).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Python hash map' })).toBeDisabled()
  expect(props.onNew).not.toHaveBeenCalled()
})

it('toggles and removes documents', () => {
  const props = setup()
  expect(screen.getByLabelText('Use notes.md')).toBeChecked()
  fireEvent.click(screen.getByLabelText('Use notes.md'))
  expect(props.onToggleDoc).toHaveBeenCalledWith('d1')
  fireEvent.click(screen.getByRole('button', { name: 'Remove notes.md' }))
  expect(props.onRemoveDoc).toHaveBeenCalledWith('d1')
})

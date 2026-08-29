import { useEffect, useRef, useState } from 'react'
import ModelSelector from './ModelSelector'
import ToolsMenu from './ToolsMenu'

function ChatComposer({ onSubmit, loading, onPipeline }) {
  const [question, setQuestion] = useState('')
  const [toolsOpen, setToolsOpen] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState('✨ Auto Gate')
  const toolsRef = useRef(null)

  useEffect(() => {
    const closeOnOutside = (event) => {
      if (toolsRef.current && !toolsRef.current.contains(event.target)) setToolsOpen(false)
    }
    const closeOnEscape = (event) => { if (event.key === 'Escape') setToolsOpen(false) }
    document.addEventListener('mousedown', closeOnOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => { document.removeEventListener('mousedown', closeOnOutside); document.removeEventListener('keydown', closeOnEscape) }
  }, [])

  const submit = () => {
    const value = question.trim()
    if (!value || loading) return
    onSubmit(value)
    setQuestion('')
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-wrap">
      <div className="composer">
        <div className="composer-plus-wrapper" ref={toolsRef}>
          <button className="composer-icon" type="button" onClick={() => setToolsOpen((open) => !open)} aria-expanded={toolsOpen} aria-label="Open tools menu">+</button>
          {toolsOpen ? <ToolsMenu onPipeline={() => { onPipeline(); setToolsOpen(false) }} /> : null}
        </div>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask GateOrchestra anything..."
          rows="1"
          disabled={loading}
        />
        <ModelSelector value={selectedStrategy} onChange={setSelectedStrategy} />
        <button className="send-button" type="button" onClick={submit} disabled={loading || !question.trim()}>
          {loading ? '...' : 'Send'}
        </button>
      </div>
      <p className="composer-note">GateOrchestra can make mistakes. Check important answers.</p>
    </div>
  )
}

export default ChatComposer
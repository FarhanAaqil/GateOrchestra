import { useState } from 'react'

const PRESETS = [
  { label: 'Arithmetic ⚡', q: 'Convert 72 kilometers per hour to meters per second.', gt: '20 m/s' },
  { label: 'Multi-Hop Bridge 🌉', q: 'What is the capital of the country where Marie Curie was born?', gt: 'Warsaw' },
  { label: 'Hard Multi-Hop 🧠', q: 'What is the official name of the country whose highest mountain shares its name with the person who surveyed it?', gt: 'Republic of India' },
  { label: 'Probability 🎲', q: 'A box contains 4 red balls, 3 blue balls, and 5 green balls. What is the probability of randomly picking a blue ball?', gt: '1/4 (or 25%)' },
]

function ChatComposer({ onSubmit, loading }) {
  const [question, setQuestion] = useState('')
  const [selectedStrategy, setSelectedStrategy] = useState('GateOrchestra')
  const [currentGt, setCurrentGt] = useState(null)

  const handlePreset = (preset) => {
    setQuestion(preset.q)
    setCurrentGt(preset.gt)
  }

  const submit = () => {
    const value = question.trim()
    if (!value || loading) return
    onSubmit(value, selectedStrategy, null, currentGt)
    setQuestion('')
    setCurrentGt(null)
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer-container">
      {/* Presets Row */}
      <div className="presets-row">
        <span style={{ fontSize: '0.72rem', fontWeight: 700, color: 'var(--c-navy)', whiteSpace: 'nowrap' }}>
          Presets:
        </span>
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className="preset-chip-btn"
            onClick={() => handlePreset(p)}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Input Textarea */}
      <div className="composer-input-row">
        <textarea
          className="composer-textarea"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask GateOrchestra or paste a multi-step reasoning problem..."
          rows={2}
          disabled={loading}
        />
      </div>

      {/* Controls & Submit */}
      <div className="composer-controls-row">
        <div className="composer-controls-left">
          <label style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--c-navy)' }}>
            Method:
          </label>
          <select
            className="method-select-input"
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
            disabled={loading}
          >
            <option value="GateOrchestra">⚡ GateOrchestra (Trained GBT)</option>
            <option value="CoT-SC">💡 CoT-SC-only (Single Agent)</option>
            <option value="Always-MAS">🤝 Always-MAS (Full Orchestrator)</option>
            <option value="RuleBasedGate">📋 Rule-Based Gate</option>
            <option value="RandomGate">🎲 Random Gate</option>
          </select>

          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Model: Groq / qwen3.8-27b • k=3
          </span>
        </div>

        <button
          className="btn-primary-ember"
          type="button"
          onClick={submit}
          disabled={loading || !question.trim()}
        >
          <span>{loading ? '⏳' : '▶'}</span>
          <span>{loading ? 'Evaluating...' : 'Run Execution'}</span>
        </button>
      </div>
    </div>
  )
}

export default ChatComposer

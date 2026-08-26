import { useState } from 'react'

const defaultForm = {
  task_id: 'test-001',
  question: 'What is machine learning?',
  context:
    'Machine learning is a branch of artificial intelligence that enables systems to learn from data.',
  ground_truth: 'Machine learning enables computers to learn patterns from data.',
  method: 'RuleBasedGate',
  k: 3,
}

function RunPanel({ onSubmit, loading, error }) {
  const [formData, setFormData] = useState(defaultForm)

  const handleChange = (event) => {
    const { name, value } = event.target
    setFormData((current) => ({
      ...current,
      [name]: name === 'k' ? Number(value) : value,
    }))
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit(formData)
  }

  return (
    <section className="panel run-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Task execution</p>
          <h2>Run orchestration</h2>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="run-form">
        <div className="field-grid">
          <label>
            <span>Task ID</span>
            <input
              name="task_id"
              value={formData.task_id}
              onChange={handleChange}
              placeholder="task-001"
            />
          </label>

          <label>
            <span>Method</span>
            <select name="method" value={formData.method} onChange={handleChange}>
              <option value="RuleBasedGate">RuleBasedGate</option>
              <option value="RandomGate">RandomGate</option>
            </select>
          </label>

          <label className="full-width">
            <span>Question</span>
            <textarea
              name="question"
              value={formData.question}
              onChange={handleChange}
              rows="3"
            />
          </label>

          <label className="full-width">
            <span>Context</span>
            <textarea
              name="context"
              value={formData.context}
              onChange={handleChange}
              rows="4"
            />
          </label>

          <label className="full-width">
            <span>Ground Truth</span>
            <textarea
              name="ground_truth"
              value={formData.ground_truth}
              onChange={handleChange}
              rows="3"
            />
          </label>

          <label>
            <span>k value</span>
            <input
              type="number"
              name="k"
              min="1"
              value={formData.k}
              onChange={handleChange}
            />
          </label>
        </div>

        {error ? <div className="api-error">{error}</div> : null}

        <button className="primary-button" type="submit" disabled={loading}>
          {loading ? 'Running…' : 'Run orchestration'}
        </button>
      </form>
    </section>
  )
}

export default RunPanel

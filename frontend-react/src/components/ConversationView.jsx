import ExecutionDetails from './ExecutionDetails'

function ConversationView({ messages, loading }) {
  if (!messages.length && !loading) {
    return (
      <div className="welcome-box">
        <div className="brand-icon" style={{ margin: '0 auto 14px', width: '48px', height: '48px', fontSize: '20px' }}>
          GO
        </div>
        <h2>GateOrchestra Precision Console</h2>
        <p>
          Adaptive learned pre-execution gating and dynamic token budgeting for multi-agent LLM systems.
          Run a reasoning query to observe real-time 8-dimensional signal extraction, GBT classification,
          and token-budgeted MAS orchestration.
        </p>
      </div>
    )
  }

  return (
    <div className="conversation-scroll-area">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`message-card ${message.role === 'user' ? 'user' : 'assistant'}`}
        >
          <div className="message-header-row">
            <div className="message-sender">
              <span className={`avatar-badge ${message.role === 'user' ? 'user' : ''}`}>
                {message.role === 'user' ? 'U' : 'GO'}
              </span>
              <span>{message.role === 'user' ? 'Operator Query' : 'GateOrchestra Resolution'}</span>
            </div>

            {message.result ? (
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.68rem',
                  color: 'var(--text-muted)',
                }}
              >
                {message.result.method || 'GateOrchestra'}
              </span>
            ) : null}
          </div>

          <div className="message-body">{message.content}</div>

          {message.role === 'assistant' && message.result ? (
            <ExecutionDetails result={message.result} />
          ) : null}
        </div>
      ))}

      {loading ? (
        <div className="message-card assistant">
          <div className="message-sender">
            <span className="avatar-badge">GO</span>
            <span>Running Probe &amp; Evaluating Gate Decision...</span>
          </div>
          <div className="typing-indicator" aria-label="GateOrchestra is evaluating">
            <span /><span /><span />
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default ConversationView

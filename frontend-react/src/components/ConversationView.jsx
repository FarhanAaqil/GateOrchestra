import ChatMessage from './ChatMessage'
import ExecutionDetails from './ExecutionDetails'

function ConversationView({ messages, loading }) {
  if (!messages.length && !loading) {
    return (
      <div className="welcome-state">
        <div className="welcome-mark">GO</div>
        <h1>GateOrchestra</h1>
        <p>What would you like to solve?</p>
      </div>
    )
  }

  return (
    <div className="conversation-list">
      {messages.map((message) => (
        <ChatMessage key={message.id} role={message.role}>
          {message.content}
          {message.role === 'assistant' && message.result ? (
            <ExecutionDetails result={message.result} />
          ) : null}
        </ChatMessage>
      ))}
      {loading ? (
        <ChatMessage role="assistant">
          <div className="typing-indicator" aria-label="GateOrchestra is working">
            <span /><span /><span />
          </div>
        </ChatMessage>
      ) : null}
    </div>
  )
}

export default ConversationView
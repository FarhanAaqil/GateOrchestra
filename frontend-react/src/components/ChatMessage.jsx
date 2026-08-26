function ChatMessage({ role, children }) {
  const isUser = role === 'user'

  return (
    <article className={`chat-message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className={`message-avatar ${isUser ? 'user-avatar' : 'assistant-avatar'}`}>
        {isUser ? 'U' : 'GO'}
      </div>
      <div className="message-content">
        <div className="message-author">{isUser ? 'You' : 'GateOrchestra'}</div>
        <div className="message-body">{children}</div>
      </div>
    </article>
  )
}

export default ChatMessage
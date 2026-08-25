function ToolsMenu({ onPipeline }) {
  return (
    <div className="tools-menu">
      <p className="tools-title">Add to chat</p>
      <button type="button" onClick={onPipeline}>🧠 Agent Pipeline</button>
      <button type="button">🔌 Plugins <small>Coming soon</small></button>
      <button type="button">📚 Library <small>Coming soon</small></button>
      <button type="button">📊 Analyze Data <small>Coming soon</small></button>
      <button type="button">📄 Document <small>Coming soon</small></button>
      <button type="button">🌐 Web Research <small>Coming soon</small></button>
      <button type="button">🖼️ Image <small>Coming soon</small></button>
    </div>
  )
}

export default ToolsMenu
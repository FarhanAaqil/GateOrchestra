function Navbar({ apiOnline, onNewExecution }) {
  return (
    <header className="top-nav">
      <div className="top-nav-left">
        <div className="brand-badge">
          <div className="brand-icon">GO</div>
          <div className="brand-text-col">
            <span className="brand-title">GateOrchestra</span>
            <span className="brand-sub">Telemetry &amp; Adaptive Gate Engine</span>
          </div>
        </div>

        <div className="gate-active-badge">
          <span>⚡</span>
          <span>AI GATE ACTIVE</span>
        </div>
      </div>

      <div className="top-nav-right">
        <div className="status-pill-telemetry">
          <span
            className={`status-indicator-dot ${
              apiOnline === true ? 'pulse' : apiOnline === false ? 'offline' : ''
            }`}
          />
          <span>{apiOnline === true ? 'Groq (qwen3.8-27b)' : apiOnline === false ? 'Backend Offline' : 'Connecting...'}</span>
          <span style={{ color: 'var(--text-muted)' }}>• 12ms</span>
        </div>

        <div className="metric-pill-dark">
          <span className="label">α:</span>
          <span className="val-lemon">1.42</span>
          <span style={{ color: '#545E76' }}>|</span>
          <span className="label">Swarms:</span>
          <span className="val-cyan">16 Active</span>
        </div>

        <button
          className="btn-primary-ember"
          type="button"
          onClick={onNewExecution}
          title="New Orchestration Task"
        >
          <span>▶</span>
          <span>Run Task</span>
        </button>
      </div>
    </header>
  )
}

export default Navbar

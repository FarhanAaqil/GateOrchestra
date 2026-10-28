function Navbar({ apiOnline, modelLabel, onNewExecution }) {
  return (
    <header className="top-nav">
      <div className="top-nav-left">
        <div className="brand-badge">
          <div className="brand-icon">GO</div>
          <div className="brand-text-col">
            <span className="brand-title">GateOrchestra</span>
            <span className="brand-sub">Pre-execution gate for multi-agent orchestration</span>
          </div>
        </div>

      </div>

      <div className="top-nav-right">
        <div className="status-pill-telemetry">
          <span
            className={`status-indicator-dot ${
              apiOnline === true ? 'pulse' : apiOnline === false ? 'offline' : ''
            }`}
          />
          <span>
            {apiOnline === true
              ? modelLabel || 'Backend online'
              : apiOnline === false
                ? 'Backend Offline'
                : 'Connecting...'}
          </span>
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

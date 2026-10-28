function Sidebar({
  conversations,
  activeId,
  view,
  collapsed,
  onToggle,
  onNewChat,
  onOpenConversation,
  onDelete,
  onNavigate,
  modelLabel,
  gateInfo,
}) {
  const navItems = [
    { id: 'chat', label: 'Orchestration Studio', icon: '⚡', badge: 'LIVE' },
    { id: 'benchmarks', label: 'Benchmarks', icon: '📊', badge: null },
    { id: 'evolution', label: 'MAS Routing', icon: '🧬', badge: null },
    { id: 'schedules', label: 'Budget Policies (k)', icon: '⏱️', badge: null },
  ]

  return (
    <aside className={`sidebar-rail ${collapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-section-top">
        <div className="sidebar-toggle-row">
          {!collapsed ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '1.1rem' }}>🎛️</span>
              <div>
                <p style={{ fontWeight: 800, fontSize: '0.85rem', color: 'var(--c-navy)' }}>Console Rail</p>
                <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  GateOrchestra
                </p>
              </div>
            </div>
          ) : null}

          <button
            className="sidebar-toggle-btn"
            type="button"
            onClick={onToggle}
            aria-label="Toggle rail"
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {collapsed ? '▶' : '◀'}
          </button>
        </div>

        <button
          className="btn-primary-ember"
          type="button"
          onClick={onNewChat}
          style={{ width: '100%', justifyContent: 'center', padding: '8px 12px' }}
        >
          <span>+</span>
          {!collapsed ? <span>New Task</span> : null}
        </button>

        <nav className="nav-group" aria-label="Console Navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-tab-btn ${view === item.id ? 'active' : ''}`}
              type="button"
              onClick={() => onNavigate(item.id)}
              title={item.label}
            >
              <div className="nav-tab-left">
                <span className="nav-tab-icon">{item.icon}</span>
                {!collapsed ? <span>{item.label}</span> : null}
              </div>
              {!collapsed && item.badge ? (
                <span className={`nav-badge-pill ${item.badge === 'LIVE' ? 'live' : ''}`}>
                  {item.badge}
                </span>
              ) : null}
            </button>
          ))}
        </nav>

        {!collapsed ? (
          <div className="sidebar-history">
            <p className="sidebar-label">Recent Tasks</p>
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`chat-history-item ${conv.id === activeId ? 'active' : ''}`}
                onClick={() => onOpenConversation(conv.id)}
                style={{ cursor: 'pointer' }}
              >
                <span className="chat-history-title">{conv.title || 'Untitled task'}</span>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(conv.id)
                  }}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-muted)',
                    fontSize: '11px',
                    padding: '2px 4px',
                  }}
                  title="Delete"
                >
                  ✕
                </button>
              </div>
            ))}
            {!conversations.length ? (
              <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', padding: '4px' }}>
                No active task traces.
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      {!collapsed && modelLabel ? (
        <div className="sidebar-resource-card">
          <div className="resource-row">
            <span>Model</span>
            <span className="resource-val">{modelLabel}</span>
          </div>
          <div className="resource-row">
            <span>Gate</span>
            <span className="resource-val" title={gateInfo?.path || ''}>
              {gateInfo
                ? `${gateInfo.classifier}${gateInfo.seed != null ? ` · seed ${gateInfo.seed}` : ''}${gateInfo.degenerate ? ' · degenerate' : ''}`
                : 'none trained'}
            </span>
          </div>
        </div>
      ) : null}
    </aside>
  )
}

export default Sidebar

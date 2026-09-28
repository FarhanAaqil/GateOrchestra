function Evolution() {
  const agents = [
    { name: 'ReAct Agent', role: 'Step-by-step reasoning & tool execution', ucb: '0.884', status: 'Primary (Highest Reward)', alloc: '90.6%', color: 'var(--c-wine)' },
    { name: 'Debate Swarm', role: 'Multi-agent adversarial consensus', ucb: '0.741', status: 'Active (Tight Budget Penalty)', alloc: '0.0%', color: 'var(--c-navy)' },
    { name: 'Reflexion Agent', role: 'Verbal self-reflection & backtrack memory', ucb: '0.812', status: 'Secondary (Error Recovery)', alloc: '9.4%', color: 'var(--c-tangerine)' },
  ]

  const kValues = [
    { k: 'k=2', budget: '2x probe tokens', savings: '+66.1%', status: 'Constrained' },
    { k: 'k=3', budget: '3x probe tokens (Default)', savings: '+78.3%', status: 'Pareto Optimal Knee' },
    { k: 'k=5', budget: '5x probe tokens', savings: '+87.0%', status: 'Permissive' },
  ]

  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Multi-Agent Swarm Telemetry</p>
        <h2>Agent Health &amp; Bandit Allocation</h2>
        <p>
          Contextual bandit (LinUCB) routing dynamics and dynamic token budget clamping across specialized sub-agents.
        </p>
      </div>

      <div className="kpi-grid">
        <article className="kpi-card">
          <span className="kpi-label">Swarm Health</span>
          <div className="kpi-value cyan">98.4%</div>
          <p className="kpi-sub">All sub-agents responsive</p>
        </article>
        <article className="kpi-card">
          <span className="kpi-label">Gate Model</span>
          <div className="kpi-value">GBTGate</div>
          <p className="kpi-sub">8 Features • 1.80 ms CPU latency</p>
        </article>
        <article className="kpi-card">
          <span className="kpi-label">LinUCB Exploration α</span>
          <div className="kpi-value savings">1.42</div>
          <p className="kpi-sub">Token-normalized reward formula</p>
        </article>
        <article className="kpi-card">
          <span className="kpi-label">Default Budget Cap</span>
          <div className="kpi-value">k = 3</div>
          <p className="kpi-sub">Dynamic clamping: B = 3 × probe_tokens</p>
        </article>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Specialized Sub-Agent Status &amp; LinUCB Allocation</h3>
          <span className="best-indicator">Active Engine: Groq / qwen3.8-27b</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          {agents.map((ag) => (
            <div
              key={ag.name}
              style={{
                backgroundColor: 'var(--c-softwhite)',
                border: '1px solid var(--c-sky)',
                borderRadius: '10px',
                padding: '16px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong style={{ color: 'var(--c-navy)', fontSize: '0.95rem' }}>{ag.name}</strong>
                <span
                  style={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid var(--c-sky)',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    fontWeight: 700,
                    color: ag.color,
                  }}
                >
                  UCB: {ag.ucb}
                </span>
              </div>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{ag.role}</p>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginTop: '4px' }}>
                <span>Status: <strong>{ag.status}</strong></span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>Allocation: <strong>{ag.alloc}</strong></span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Dynamic Token Budget Clamping Sweep (k ∈ &#123;2, 3, 5&#125;)</h3>
        </div>

        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head" style={{ gridTemplateColumns: '1fr 2fr 1.5fr 1.5fr' }}>
            <span>Multiplier</span>
            <span>Budget Ceiling Rule</span>
            <span>Token Savings</span>
            <span>Operating Regime</span>
          </div>
          {kValues.map((row) => (
            <div
              key={row.k}
              className={`benchmark-row ${row.k === 'k=3' ? 'best-row' : ''}`}
              style={{ gridTemplateColumns: '1fr 2fr 1.5fr 1.5fr' }}
            >
              <strong>{row.k}</strong>
              <span>{row.budget}</span>
              <span style={{ color: 'var(--c-ember)', fontWeight: 700 }}>{row.savings}</span>
              <span>{row.status}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default Evolution

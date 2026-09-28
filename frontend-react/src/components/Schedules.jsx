const policies = [
  ['Default Clamping Policy', 'All Ingestion', 'k = 3', 'B = 3 × probe_tokens', 'Enforced'],
  ['High-Precision Safety Sweep', 'Complex Multi-Hop', 'k = 5', 'B = 5 × probe_tokens', 'Active'],
  ['Low-Budget Fast-Path', 'Arithmetic / Direct', 'k = 2', 'B = 2 × probe_tokens', 'Active'],
  ['Single-Agent CoT-SC Floor', 'Simple Query Pass', 'k = 0', 'Fast STOP (0 MAS Tokens)', 'Enforced'],
]

function Schedules() {
  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Budget Governance</p>
        <h2>Token Budget Policies</h2>
        <p>Pre-execution budget clamping and maximum MAS token allocation rules.</p>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Active Orchestration Policies</h3>
          <span className="best-indicator">Enforcement: Strict Hard-Cap</span>
        </div>

        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head">
            <span>Policy Name</span>
            <span>Domain Scope</span>
            <span>Multiplier</span>
            <span>Clamping Rule</span>
            <span>Status</span>
          </div>
          {policies.map((row) => (
            <div className="benchmark-row" key={row[0]}>
              <strong>{row[0]}</strong>
              <span>{row[1]}</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--c-navy)', fontWeight: 700 }}>{row[2]}</span>
              <span>{row[3]}</span>
              <span style={{ color: 'var(--c-ember)', fontWeight: 700 }}>{row[4]}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default Schedules

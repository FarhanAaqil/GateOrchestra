const kValues = [
  ['k = 2', 'MAS budget = 2 × the task\'s probe tokens'],
  ['k = 3', 'MAS budget = 3 × the task\'s probe tokens (default)'],
  ['k = 5', 'MAS budget = 5 × the task\'s probe tokens'],
]

const rules = [
  ['Selection', 'One k is chosen per benchmark run by validation calibration and applied to every task.'],
  ['Equal budgets', 'Always-MAS and every gated escalation use the same k × probe-token cap for a task.'],
  ['Enforcement', 'Before each LLM call the completion cap is the remaining budget minus an estimate of the prompt; no call is made once fewer than 32 tokens fit.'],
  ['Accounting', 'Recorded spend is the provider\'s true prompt + completion total. It is never clipped, so it can exceed the cap by the prompt-estimate error.'],
  ['STOP', 'A STOP decision spends no MAS tokens; the probe answer is returned.'],
]

function Schedules() {
  return (
    <div className="research-view">
      <div className="research-heading">
        <p className="header-kicker">Budget Governance</p>
        <h2>Token Budget Policy</h2>
        <p>How the escalation budget is set, enforced, and accounted for.</p>
      </div>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Multiplier Sweep (k ∈ &#123;2, 3, 5&#125;)</h3>
        </div>
        <div className="benchmark-table">
          <div className="benchmark-row benchmark-head" style={{ gridTemplateColumns: '1fr 3fr' }}>
            <span>Multiplier</span>
            <span>Budget rule</span>
          </div>
          {kValues.map(([k, rule]) => (
            <div className="benchmark-row" key={k} style={{ gridTemplateColumns: '1fr 3fr' }}>
              <strong style={{ fontFamily: 'var(--font-mono)' }}>{k}</strong>
              <span>{rule}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="research-section">
        <div className="section-title-row">
          <h3>Rules</h3>
        </div>
        <div className="benchmark-table">
          {rules.map(([name, text]) => (
            <div className="benchmark-row" key={name} style={{ gridTemplateColumns: '1fr 3fr' }}>
              <strong>{name}</strong>
              <span>{text}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

export default Schedules

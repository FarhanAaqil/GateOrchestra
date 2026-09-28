function StatCard({ label, value, detail, type = 'default', badge }) {
  let valueClass = 'kpi-value'
  if (type === 'savings') valueClass += ' savings'
  if (type === 'cyan') valueClass += ' cyan'

  return (
    <article className="kpi-card">
      <div className="kpi-header">
        <span className="kpi-label">{label}</span>
        {badge ? (
          <span
            className="kpi-tag"
            style={{
              backgroundColor: type === 'savings' ? 'var(--c-lemon)' : 'var(--c-ice)',
              color: 'var(--c-navy)',
              border: '1px solid var(--c-sky)',
            }}
          >
            {badge}
          </span>
        ) : null}
      </div>
      <div className={valueClass}>{value}</div>
      {detail ? <p className="kpi-sub">{detail}</p> : null}
    </article>
  )
}

export default StatCard

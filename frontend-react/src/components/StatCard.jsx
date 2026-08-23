function StatCard({ label, value, accent, detail }) {
  return (
    <article className="stat-card">
      <div className="stat-header">
        <span className="card-label">{label}</span>
        {accent ? <span className="accent-dot" aria-hidden="true" /> : null}
      </div>
      <div className="stat-value">{value}</div>
      {detail ? <p className="stat-detail">{detail}</p> : null}
    </article>
  )
}

export default StatCard

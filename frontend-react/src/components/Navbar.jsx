function Navbar() {
  return (
    <header className="topbar">
      <div className="brand-group">
        <div className="brand-mark">GO</div>
        <div>
          <p className="eyebrow">Research orchestration</p>
          <h1>GateOrchestra</h1>
        </div>
      </div>

      <div className="status-pill">
        <span className="status-dot" aria-hidden="true" />
        API online
      </div>
    </header>
  )
}

export default Navbar

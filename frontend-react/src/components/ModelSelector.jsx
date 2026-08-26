import { useState } from 'react'

const strategies = ['✨ Auto Gate', '🧠 CoT-SC', '🤝 Always-MAS', '🎲 Random Gate', '📐 Rule-Based Gate', '⚡ GateOrchestra']

function ModelSelector({ value, onChange }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="selector-wrap">
      <button className="selector-button" type="button" onClick={() => setOpen((current) => !current)}>
        <span>Model:</span> {value} <span className="select-chevron">v</span>
      </button>
      {open ? <div className="selector-popover model-popover">{strategies.map((strategy) => <button key={strategy} className={strategy === value ? 'selected' : ''} type="button" onClick={() => { onChange(strategy); setOpen(false) }}>{strategy}</button>)}</div> : null}
    </div>
  )
}

export default ModelSelector
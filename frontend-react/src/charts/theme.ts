// Shared Recharts styling: muted navy axes, faint solid gridlines, navy tooltips.
export const AXIS = { stroke: 'rgba(27,38,59,0.55)', fontSize: 11, fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }
export const GRID = { stroke: 'rgba(27,38,59,0.08)', strokeDasharray: '' }
export const TOOLTIP = {
  contentStyle: {
    background: '#1B263B',
    border: '1px solid #1B263B',
    borderRadius: 8,
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
    fontSize: 12,
    color: '#F1FAFF',
    boxShadow: '0 8px 24px -12px rgba(27,38,59,0.5)',
  },
  labelStyle: { color: '#A8DBFF', fontWeight: 600 },
  itemStyle: { color: '#F1FAFF' },
}

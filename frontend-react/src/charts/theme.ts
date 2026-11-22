// Shared Recharts styling: muted forest axes, faint solid gridlines, forest tooltips.
export const AXIS = { stroke: 'rgba(19,52,42,0.55)', fontSize: 11, fontFamily: 'Instrument Sans, ui-sans-serif, system-ui, sans-serif' }
export const GRID = { stroke: 'rgba(19,52,42,0.08)', strokeDasharray: '' }
export const TOOLTIP = {
  contentStyle: {
    background: '#13342A',
    border: '1px solid #13342A',
    borderRadius: 8,
    fontFamily: 'Instrument Sans, ui-sans-serif, system-ui, sans-serif',
    fontSize: 12,
    color: '#FFFCF7',
    boxShadow: '0 8px 24px -12px rgba(19,52,42,0.5)',
  },
  labelStyle: { color: '#B7D9C4', fontWeight: 600 },
  itemStyle: { color: '#FFFCF7' },
}

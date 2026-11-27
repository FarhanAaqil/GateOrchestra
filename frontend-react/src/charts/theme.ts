// Shared Recharts styling: muted teal axes, hairline gridlines, and a white
// tooltip card that matches the panels around it.
export const AXIS = { stroke: '#6B6B6B', fontSize: 11, fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif' }
export const GRID = { stroke: '#E6E8E6', strokeDasharray: '' }
export const TOOLTIP = {
  contentStyle: {
    background: '#FFFFFF',
    border: '1px solid #E6E8E6',
    borderRadius: 12,
    fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
    fontSize: 12,
    color: '#1A1D1C',
    boxShadow: '0 10px 30px -18px rgba(11,74,107,0.45)',
    padding: '8px 12px',
  },
  labelStyle: { color: '#0B4A6B', fontWeight: 600, marginBottom: 2 },
  itemStyle: { color: '#1A1D1C' },
}

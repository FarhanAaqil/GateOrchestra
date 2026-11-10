/**
 * Zoom the accuracy axis to the data instead of always showing 0-100%.
 * When every method scores above 80% a full-height axis squeezes all points into the top
 * strip; the zoomed axis keeps 5 points of headroom and stays on whole-number ticks.
 */
export function accuracyDomain(values: number[]): [number, number] {
  if (!values.length) return [0, 100]
  const low = Math.min(...values)
  if (low < 50) return [0, 100]
  return [Math.max(0, Math.floor((low - 5) / 5) * 5), 100]
}

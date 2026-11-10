/**
 * Live runs get ids like "ui-1794080322070" (the number is the start time in milliseconds) or
 * "live_demo_1". Show a readable label and keep the raw id for a tooltip.
 */
export function friendlyRunLabel(taskId: string): string {
  const ui = taskId.match(/^ui-(\d{12,14})$/)
  if (ui) {
    const when = new Date(Number(ui[1]))
    if (!Number.isNaN(when.getTime())) {
      const day = when.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
      const time = when.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
      return `Run on ${day} at ${time}`
    }
  }
  const demo = taskId.match(/^live_demo_(\d+)$/)
  if (demo) return `Demo run ${demo[1]}`
  return taskId
}

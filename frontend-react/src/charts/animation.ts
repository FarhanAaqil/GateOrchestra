import { useReducedMotion } from 'framer-motion'

/** Chart series animate unless the OS asks for reduced motion. */
export function useChartAnimation(): boolean {
  return !useReducedMotion()
}

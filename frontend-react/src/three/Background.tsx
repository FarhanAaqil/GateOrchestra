import { lazy, Suspense, useEffect, useState } from 'react'
import { useApp } from '../lib/context'

const NeuralField = lazy(() => import('./NeuralField'))

/**
 * The page ground: ruled paper. Two hairline grids at a few per cent opacity,
 * nothing else. No colour fields, no glow — the cards and the type carry the
 * design, and a quiet ground keeps long tables and charts readable.
 */
function FlatBackdrop({ drift }: { drift: boolean }) {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden>
      <div
        className="absolute inset-[-60px]"
        style={{
          backgroundImage:
            'linear-gradient(to right, rgba(11,74,107,0.045) 1px, transparent 1px),' +
            'linear-gradient(to bottom, rgba(11,74,107,0.045) 1px, transparent 1px)',
          backgroundSize: '96px 96px',
          animation: drift ? 'drift 40s linear infinite alternate' : undefined,
        }}
      />
    </div>
  )
}

/** True once the browser is idle after the first paint, so decoration never delays content. */
function useIdle(): boolean {
  const [idle, setIdle] = useState(false)
  useEffect(() => {
    if ('requestIdleCallback' in window) {
      const id = window.requestIdleCallback(() => setIdle(true), { timeout: 2500 })
      return () => window.cancelIdleCallback(id)
    }
    const id = setTimeout(() => setIdle(true), 1500)
    return () => clearTimeout(id)
  }, [])
  return idle
}

export default function Background() {
  const { use3D, reducedMotion, isMobile } = useApp()
  const idle = useIdle()
  return (
    <>
      <FlatBackdrop drift={!reducedMotion && !isMobile} />
      {use3D && idle ? (
        <Suspense fallback={null}>
          <NeuralField />
        </Suspense>
      ) : null}
    </>
  )
}

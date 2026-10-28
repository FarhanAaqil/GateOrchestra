import { lazy, Suspense } from 'react'
import { useApp } from '../lib/context'

const NeuralField = lazy(() => import('./NeuralField'))

/** Soft colour fields (sky, lemon, ember) plus a slow-drifting navy dot grid. */
function FlatBackdrop({ drift }: { drift: boolean }) {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10 overflow-hidden" aria-hidden>
      <div className="absolute -left-40 top-[-15%] h-[560px] w-[560px] rounded-full bg-sky/50 blur-3xl" />
      <div className="absolute right-[-12%] top-[30%] h-[460px] w-[460px] rounded-full bg-lemon/40 blur-3xl" />
      <div className="absolute bottom-[-25%] left-[35%] h-[420px] w-[420px] rounded-full bg-ember/15 blur-3xl" />
      <div
        className="absolute inset-[-60px] opacity-60"
        style={{
          backgroundImage: 'radial-gradient(rgba(27,38,59,0.13) 1px, transparent 1px)',
          backgroundSize: '34px 34px',
          animation: drift ? 'drift 40s linear infinite alternate' : undefined,
        }}
      />
    </div>
  )
}

export default function Background() {
  const { use3D, reducedMotion } = useApp()
  return (
    <>
      <FlatBackdrop drift={!reducedMotion} />
      {use3D ? (
        <Suspense fallback={null}>
          <NeuralField />
        </Suspense>
      ) : null}
    </>
  )
}

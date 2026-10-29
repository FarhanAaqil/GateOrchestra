import { lazy, Suspense } from 'react'
import { useApp } from '../lib/context'
import GateScene2D from './GateScene2D'
import type { GateRunInput } from './useGateRun'

const GateScene = lazy(() => import('./GateScene'))

/** The gate scene: full 3D when allowed, otherwise the SVG version. */
export default function GateView(props: GateRunInput & { className?: string }) {
  const { use3D, reducedMotion } = useApp()
  const { className = 'h-[340px] md:h-[420px]', ...run } = props
  const fallback = <GateScene2D {...run} animate={!reducedMotion} />
  return (
    <div className={`relative w-full ${className}`}>
      {use3D ? <Suspense fallback={fallback}><GateScene {...run} /></Suspense> : fallback}
    </div>
  )
}

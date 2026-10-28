import { Line } from '@react-three/drei'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Bloom, EffectComposer } from '@react-three/postprocessing'
import { useCallback, useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { STRATEGY_LABELS } from '../lib/format'
import { agentPosition, LAYOUT, orbPosition, STRATEGIES, type Vec3 } from './gateTimeline'
import { useGateRun, type GateRunInput } from './useGateRun'

// Palette: cyan = gate, sky = CoT-SC path, ember/wine = multi-agent path, lemon = chosen agent.
const CYAN = new THREE.Color('#00E5FF')
const SKY = new THREE.Color('#A8DBFF')
const EMBER = new THREE.Color('#FF6B35')
const WINE = new THREE.Color('#8B1538')
const LEMON = new THREE.Color('#FFF56B')
const NAVY = '#1B263B'

// Labels are plain DOM nodes over the canvas, positioned each frame by
// projecting a 3D point; this avoids one React root per label.
const STATIC_LABELS: { text: string; tone: string; at: Vec3 }[] = [
  { text: 'Gate', tone: '#00E5FF', at: [LAYOUT.gate[0], LAYOUT.gate[1] + 1.3, 0] },
  { text: 'CoT-SC answer', tone: '#A8DBFF', at: [LAYOUT.cot[0], LAYOUT.cot[1] - 0.62, 0] },
  // Below the agents' orbit so the labels never collide.
  { text: 'Orchestrator', tone: '#FF6B35', at: [LAYOUT.core[0], LAYOUT.core[1] - 1.62, 0] },
]
const AGENT_LABEL_OFFSET = 0.4
// World-space extent the camera must keep in view (gate approach to the orbiting agents).
// The incoming path starts off-screen on the left by design.
const SCENE_WIDTH = 8.6
const SCENE_HEIGHT = 5.9
const SCENE_CENTER_X = 1.75
const SCENE_CENTER_Y = -0.35

function curvePoints(a: Vec3, control: Vec3, b: Vec3): Vec3[] {
  const curve = new THREE.QuadraticBezierCurve3(new THREE.Vector3(...a), new THREE.Vector3(...control), new THREE.Vector3(...b))
  return curve.getPoints(40).map((p) => [p.x, p.y, p.z] as Vec3)
}

type PlaceLabel = (index: number, x: number, y: number, opacity: number) => void

function Scene({ placeLabel: place, ...props }: GateRunInput & { placeLabel: PlaceLabel }) {
  const sample = useGateRun(props)
  const { camera, size } = useThree()
  const gateOuter = useRef<THREE.Mesh>(null)
  const gateInner = useRef<THREE.Mesh>(null)
  const core = useRef<THREE.Mesh>(null)
  const orb = useRef<THREE.Mesh>(null)
  const orbLight = useRef<THREE.PointLight>(null)
  const cotMat = useRef<THREE.MeshStandardMaterial>(null)
  const agentRefs = useRef<(THREE.Mesh | null)[]>([])
  const agentMats = useRef<(THREE.MeshStandardMaterial | null)[]>([])
  const stream = useRef<THREE.Mesh>(null)
  const scratch = useMemo(() => new THREE.Vector3(), [])

  // Pull the camera back until the whole layout fits, whatever the panel's shape.
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera
    const halfFov = THREE.MathUtils.degToRad(cam.fov / 2)
    const aspect = size.width / Math.max(size.height, 1)
    const fitWidth = SCENE_WIDTH / 2 / (Math.tan(halfFov) * aspect)
    const fitHeight = SCENE_HEIGHT / 2 / Math.tan(halfFov)
    cam.position.set(SCENE_CENTER_X, SCENE_CENTER_Y, Math.max(fitWidth, fitHeight))
    cam.lookAt(SCENE_CENTER_X, SCENE_CENTER_Y, 0)
    cam.updateProjectionMatrix()
  }, [camera, size])

  const { gate, cot, core: corePos } = LAYOUT
  const stopPath = useMemo(() => curvePoints(gate, [gate[0] + 1.2, (gate[1] + cot[1]) / 2, 0.4], cot), [gate, cot])
  const masPath = useMemo(() => curvePoints(gate, [(gate[0] + corePos[0]) / 2, gate[1] + 1.1, 0.3], corePos), [gate, corePos])
  const inPath = useMemo(() => [LAYOUT.start, gate] as Vec3[], [gate])

  const placeLabel = (index: number, at: Vec3, opacity = 1) => {
    scratch.set(...at).project(camera)
    place(index, ((scratch.x + 1) / 2) * size.width, ((1 - scratch.y) / 2) * size.height, opacity)
  }

  useFrame((state, delta) => {
    const clock = state.clock.elapsedTime
    const run = sample(performance.now() / 1000)
    gateOuter.current?.rotateZ(delta * 0.45)
    gateInner.current?.rotateZ(-delta * 0.8)
    core.current?.rotateY(delta * 0.5)
    core.current?.rotateX(delta * 0.2)

    const { position, visible, atTarget } = orbPosition(run.phase, run.elapsed, run.decision, run.strategy, clock)
    if (orb.current) {
      orb.current.visible = visible
      orb.current.position.set(...position)
      orb.current.scale.setScalar(run.phase === 'waiting' ? 1 + Math.sin(clock * 8) * 0.18 : 1)
    }
    if (orbLight.current) {
      orbLight.current.visible = visible
      orbLight.current.position.set(...position)
    }

    const stopLit = run.phase === 'route' && run.decision === 'STOP' && atTarget
    if (cotMat.current) cotMat.current.emissiveIntensity = THREE.MathUtils.lerp(cotMat.current.emissiveIntensity, stopLit ? 2.6 : 0.5, 0.08)

    STATIC_LABELS.forEach((label, i) => placeLabel(i, label.at))
    STRATEGIES.forEach((strategy, i) => {
      const p = agentPosition(i, clock)
      agentRefs.current[i]?.position.set(...p)
      const chosen = run.phase === 'route' && run.decision === 'ESCALATE' && run.strategy === strategy
      placeLabel(STATIC_LABELS.length + i, [p[0], p[1] + AGENT_LABEL_OFFSET, p[2]], chosen ? 1 : 0.55)
      const mat = agentMats.current[i]
      if (mat) {
        mat.emissive.copy(chosen ? LEMON : WINE)
        mat.emissiveIntensity = THREE.MathUtils.lerp(mat.emissiveIntensity, chosen && atTarget ? 2.6 : chosen ? 1.4 : 0.9, 0.1)
      }
    })

    // Beam from the core to the chosen agent while escalating.
    const index = run.strategy ? STRATEGIES.indexOf(run.strategy) : -1
    if (stream.current) {
      const show = run.phase === 'route' && run.decision === 'ESCALATE' && index >= 0 && run.elapsed > 1.2
      stream.current.visible = show
      if (show) {
        const target = new THREE.Vector3(...agentPosition(index, clock))
        const from = new THREE.Vector3(...corePos)
        stream.current.position.copy(from.clone().lerp(target, 0.5))
        stream.current.lookAt(target)
        stream.current.rotateX(Math.PI / 2)
        stream.current.scale.set(1, from.distanceTo(target), 1)
      }
    }
  })

  return (
    <>
      <color attach="background" args={[NAVY]} />
      <ambientLight intensity={0.25} />
      <pointLight position={[0, 4, 6]} intensity={30} color="#A8DBFF" />
      <pointLight ref={orbLight} intensity={8} distance={4} color="#F7F7F7" />

      <Line points={inPath} color="#00E5FF" lineWidth={1} transparent opacity={0.25} dashed dashSize={0.2} gapSize={0.15} />
      <Line points={stopPath} color="#A8DBFF" lineWidth={1.8} transparent opacity={0.6} />
      <Line points={masPath} color="#FF6B35" lineWidth={1.8} transparent opacity={0.6} />

      <group position={gate}>
        <mesh ref={gateOuter}>
          <torusGeometry args={[0.95, 0.05, 24, 120]} />
          <meshStandardMaterial color={NAVY} emissive={CYAN} emissiveIntensity={2.4} />
        </mesh>
        <mesh ref={gateInner} rotation={[0, 0, 0.6]}>
          <torusGeometry args={[0.72, 0.018, 16, 100, Math.PI * 1.6]} />
          <meshStandardMaterial color={NAVY} emissive={CYAN} emissiveIntensity={1.6} />
        </mesh>
        <mesh>
          <circleGeometry args={[0.7, 48]} />
          <meshBasicMaterial color="#00E5FF" transparent opacity={0.06} side={THREE.DoubleSide} />
        </mesh>
      </group>

      <mesh position={cot}>
        <sphereGeometry args={[0.32, 32, 32]} />
        <meshStandardMaterial ref={cotMat} color={NAVY} emissive={SKY} emissiveIntensity={0.5} />
      </mesh>

      <mesh ref={core} position={corePos}>
        <icosahedronGeometry args={[0.45, 1]} />
        <meshStandardMaterial color={NAVY} emissive={EMBER} emissiveIntensity={1.3} flatShading />
      </mesh>
      {STRATEGIES.map((strategy, i) => (
        <mesh
          key={strategy}
          ref={(el) => {
            agentRefs.current[i] = el
          }}
        >
          <sphereGeometry args={[0.18, 24, 24]} />
          <meshStandardMaterial
            ref={(el) => {
              agentMats.current[i] = el
            }}
            color={NAVY}
            emissive={WINE}
            emissiveIntensity={0.9}
          />
        </mesh>
      ))}
      <mesh ref={stream} visible={false}>
        <cylinderGeometry args={[0.02, 0.02, 1, 8]} />
        <meshBasicMaterial color="#FFF56B" transparent opacity={0.85} />
      </mesh>

      <mesh ref={orb} visible={false}>
        <sphereGeometry args={[0.16, 24, 24]} />
        <meshStandardMaterial color="#F7F7F7" emissive="#F1FAFF" emissiveIntensity={3} />
      </mesh>
    </>
  )
}

export default function GateScene(props: GateRunInput) {
  const labels = useRef<(HTMLDivElement | null)[]>([])
  const placeLabel = useCallback<PlaceLabel>((index, x, y, opacity) => {
    const el = labels.current[index]
    if (!el) return
    el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px) translate(-50%, -50%)`
    el.style.opacity = String(opacity)
  }, [])
  const labelDefs = [
    ...STATIC_LABELS.map((l) => ({ text: l.text, tone: l.tone })),
    ...STRATEGIES.map((s) => ({ text: STRATEGY_LABELS[s], tone: '#FFF56B' })),
  ]
  return (
    <div className="relative h-full w-full overflow-hidden">
      <Canvas dpr={[1, 1.75]} camera={{ position: [0, -0.3, 10.5], fov: 40 }} gl={{ antialias: true }}>
        <Scene {...props} placeLabel={placeLabel} />
        <EffectComposer>
          <Bloom intensity={1.1} luminanceThreshold={0.15} luminanceSmoothing={0.3} mipmapBlur />
        </EffectComposer>
      </Canvas>
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        {labelDefs.map((label, i) => (
          <div
            key={label.text}
            ref={(el) => {
              labels.current[i] = el
            }}
            className="absolute left-0 top-0 whitespace-nowrap rounded-full border bg-navy/80 px-2.5 py-0.5 font-mono text-[11px] backdrop-blur transition-opacity duration-300"
            style={{ borderColor: `${label.tone}66`, color: label.tone, opacity: 0 }}
          >
            {label.text}
          </div>
        ))}
      </div>
    </div>
  )
}

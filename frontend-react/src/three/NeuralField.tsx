import { Canvas, useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

const NODE_COUNT = 140
const DUST_COUNT = 900
const LINK_DISTANCE = 1.55

function seededRandom(seed: number) {
  let s = seed
  return () => {
    s = (s * 16807) % 2147483647
    return (s - 1) / 2147483646
  }
}

/** Nodes, faint links between near neighbours, and a dust layer. */
function Mesh() {
  const group = useRef<THREE.Group>(null)

  const { nodes, links, dust } = useMemo(() => {
    const rand = seededRandom(7)
    const nodePositions = new Float32Array(NODE_COUNT * 3)
    const points: THREE.Vector3[] = []
    for (let i = 0; i < NODE_COUNT; i++) {
      const p = new THREE.Vector3((rand() - 0.5) * 14, (rand() - 0.5) * 8, (rand() - 0.5) * 6 - 2)
      points.push(p)
      nodePositions.set([p.x, p.y, p.z], i * 3)
    }
    const linkPositions: number[] = []
    for (let i = 0; i < NODE_COUNT; i++) {
      for (let j = i + 1; j < NODE_COUNT; j++) {
        if (points[i].distanceTo(points[j]) < LINK_DISTANCE) {
          linkPositions.push(points[i].x, points[i].y, points[i].z, points[j].x, points[j].y, points[j].z)
        }
      }
    }
    const dustPositions = new Float32Array(DUST_COUNT * 3)
    for (let i = 0; i < DUST_COUNT; i++) {
      dustPositions.set([(rand() - 0.5) * 26, (rand() - 0.5) * 16, (rand() - 0.5) * 12 - 4], i * 3)
    }
    return { nodes: nodePositions, links: new Float32Array(linkPositions), dust: dustPositions }
  }, [])

  useFrame((state, delta) => {
    if (!group.current) return
    group.current.rotation.y += delta * 0.015
    // Gentle parallax toward the pointer.
    const targetX = state.pointer.y * 0.08
    const targetZ = -state.pointer.x * 0.05
    group.current.rotation.x += (targetX - group.current.rotation.x) * 0.02
    group.current.rotation.z += (targetZ - group.current.rotation.z) * 0.02
  })

  return (
    <group ref={group}>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[nodes, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#1B263B" size={0.05} sizeAttenuation transparent opacity={0.35} />
      </points>
      <lineSegments>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[links, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#1B263B" transparent opacity={0.07} />
      </lineSegments>
      <points>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[dust, 3]} />
        </bufferGeometry>
        <pointsMaterial color="#FF6B35" size={0.03} sizeAttenuation transparent opacity={0.35} />
      </points>
    </group>
  )
}

/** Full-screen background canvas; sits behind content and ignores pointer hits. */
export default function NeuralField() {
  return (
    <div className="pointer-events-none fixed inset-0 -z-10" aria-hidden>
      <Canvas
        dpr={[1, 1.5]}
        camera={{ position: [0, 0, 7], fov: 55 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'low-power' }}
        eventSource={document.body}
        eventPrefix="client"
      >
        {/* Drawn in navy and ember on the light Ice canvas; bloom would wash it out. */}
        <Mesh />
      </Canvas>
    </div>
  )
}

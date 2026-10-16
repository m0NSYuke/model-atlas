import { useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Html, Line } from '@react-three/drei'
import { Color, CubicBezierCurve3, InstancedMesh, Matrix4, Quaternion, Vector3, type Curve } from 'three'
import { fanouts, type ViewEdge, type ViewGraph } from '../../core/layout/viewGraph'
import { localizedNode } from '../../i18n/translations'
import { useModelStore } from '../../store/modelStore'
import { usePalette } from './theme'
import { edgeCurve } from './useViewGraph'

export interface EdgePath { edge: ViewEdge; curve: Curve<Vector3>; points: Vector3[]; color: string; length: number; label?: string; labelAtEnd?: boolean }


export function useEdgePaths(graph: ViewGraph): EdgePath[] {
  const edgeColor = usePalette().edges
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  return useMemo(() => graph.edges.flatMap((edge) => {
    const a = graph.byId.get(edge.source)
    const b = graph.byId.get(edge.target)
    if (!a || !b) return []
    const curve = edgeCurve(edge, a, b)
    const length = curve.getLength()
    const points = curve.getSpacedPoints(Math.max(12, Math.min(96, Math.round(length * 5))))
    const label = edge.stub ? typeof edge.stub.label === 'string' ? edge.stub.label : localizedNode(edge.stub.label, locale, modelId).name : undefined
    return [{ edge, curve, points, color: edgeColor[edge.type ?? 'data'], length, label }]
  }).concat(fanouts(graph).map(({ source, count, type }) => {
    // A source whose every use is a tap gets a short exit tap that says how many blocks read it.
    const a = graph.byId.get(source)!
    const start = new Vector3(a.position[0] + a.size[0] / 2, a.position[1], a.position[2])
    const end = start.clone().add(new Vector3(0.95, 0, 0))
    const curve = new CubicBezierCurve3(start, start.clone().lerp(end, 0.33), start.clone().lerp(end, 0.66), end)
    const edge: ViewEdge = { id: `${source}>taps`, source, target: source, type }
    return { edge, curve, points: curve.getSpacedPoints(12), color: edgeColor[type], length: 0.95, label: locale === 'zh' ? `→ ${count} 处` : `→ ${count} uses`, labelAtEnd: true }
  })), [graph, edgeColor, locale, modelId])
}

const up = new Vector3(0, 1, 0)

/** A single connection: a thin glowing line plus a small arrow head. */
function FlowEdge({ path, state }: { path: EdgePath; state: 'normal' | 'focus' | 'muted' }) {
  const { edge, points, curve, color } = path
  const dashed = Boolean(edge.stub) || edge.type === 'condition' || edge.type === 'training' || edge.type === 'feedback'
  const end = points[points.length - 1]
  const tangent = curve.getTangentAt(0.995).normalize()
  const quaternion = useMemo(() => new Quaternion().setFromUnitVectors(up, tangent), [tangent])
  const arrowPosition = end.clone().addScaledVector(tangent, -0.09)
  const light = usePalette().theme === 'light'
  const opacity = state === 'muted' ? (light ? 0.12 : 0.07) : state === 'focus' ? 1 : edge.type === 'data' || !edge.type ? (light ? 0.7 : 0.55) : 0.8
  const width = state === 'focus' ? 2.6 : edge.type === 'residual' || edge.type === 'branch' ? 1.5 : 1.7
  return <group>
    <Line points={points} color={color} lineWidth={width} transparent opacity={opacity} dashed={dashed} dashSize={0.22} gapSize={0.14} depthWrite={false} />
    <mesh position={arrowPosition} quaternion={quaternion}>
      <coneGeometry args={[state === 'focus' ? 0.09 : 0.07, 0.22, 10]} />
      <meshBasicMaterial color={color} transparent opacity={Math.min(1, opacity + 0.2)} depthWrite={false} />
    </mesh>
    {edge.transform && state !== 'muted' && <ReshapeMark path={path} focus={state === 'focus'} />}
    {path.label && state !== 'muted' && <Html center position={path.labelAtEnd ? points[points.length - 1].clone().add(new Vector3(0.55, 0, 0)) : points[0]} zIndexRange={[4, 0]} style={{ pointerEvents: 'none' }}>
      <span className={`stub-tag ${state === 'focus' ? 'focus' : ''} ${path.labelAtEnd ? 'out' : ''}`} style={{ ['--accent' as string]: color }}>{path.label}</span>
    </Html>}
  </group>
}

/** A reshape on the way (view, transpose, flatten, rearrange…): a small diamond mid-line, named when in focus. */
function ReshapeMark({ path, focus }: { path: EdgePath; focus: boolean }) {
  const point = useMemo(() => path.curve.getPointAt(0.5), [path.curve])
  const transform = path.edge.transform!
  return <group position={point}>
    <mesh rotation={[0, 0, Math.PI / 4]}><boxGeometry args={[0.16, 0.16, 0.16]} /><meshBasicMaterial color={path.color} transparent opacity={focus ? 1 : 0.85} depthWrite={false} /></mesh>
    {focus && <Html center position={[0, 0.32, 0]} zIndexRange={[6, 0]} style={{ pointerEvents: 'none' }}>
      <span className="reshape-tag" style={{ ['--accent' as string]: path.color }}><b>{transform.ops.map((op) => op === 'index' ? '[…]' : op).join(' · ')}</b>{transform.shape.join('×')}</span>
    </Html>}
  </group>
}

export function FlowEdges({ paths, focus, trace, path: route }: { paths: EdgePath[]; focus: Set<string> | null; trace: boolean; path?: string[] | null }) {
  // On a traced path only the consecutive hops light up.
  const hops = route ? new Set(route.slice(1).flatMap((id, index) => [`${route[index]}>${id}`, `${id}>${route[index]}`])) : null
  return <>{paths.map((path) => {
    const incident = hops ? hops.has(`${path.edge.source}>${path.edge.target}`) : focus !== null && focus.has(path.edge.source) && focus.has(path.edge.target)
    const state = incident ? 'focus' : trace && focus ? 'muted' : 'normal'
    return <FlowEdge key={path.edge.id} path={path} state={state} />
  })}</>
}

const matrix = new Matrix4()
const scaleVector = new Vector3()
const identity = new Quaternion()

/** Small bright particles travelling along every edge, showing the direction tensors flow. */
export function FlowParticles({ paths, focus, speed = 1 }: { paths: EdgePath[]; focus: Set<string> | null; speed?: number }) {
  const mesh = useRef<InstancedMesh>(null)
  const boost = usePalette().theme === 'light' ? 0.85 : 1.3
  const particles = useMemo(() => {
    const list: { path: EdgePath; offset: number }[] = []
    for (const path of paths) {
      if (focus && !(focus.has(path.edge.source) && focus.has(path.edge.target))) continue
      const count = Math.max(1, Math.min(4, Math.round(path.length / 4)))
      for (let index = 0; index < count; index++) list.push({ path, offset: index / count })
      if (list.length > 900) break
    }
    return list
  }, [paths, focus])
  useLayoutEffect(() => {
    const instanced = mesh.current
    if (!instanced) return
    const color = new Color()
    particles.forEach((particle, index) => instanced.setColorAt(index, color.set(particle.path.color).multiplyScalar(boost)))
    if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true
  }, [particles, boost])

  useFrame(({ clock }) => {
    const instanced = mesh.current
    if (!instanced) return
    const time = clock.getElapsedTime() * 0.32 * speed
    particles.forEach(({ path, offset }, index) => {
      const travel = (time * (6 / Math.max(3, path.length)) + offset) % 1
      const point = path.points[Math.min(path.points.length - 1, Math.floor(travel * (path.points.length - 1)))]
      const fade = Math.sin(travel * Math.PI)
      scaleVector.setScalar(0.35 + fade * 0.65)
      matrix.compose(point, identity, scaleVector)
      instanced.setMatrixAt(index, matrix)
    })
    instanced.instanceMatrix.needsUpdate = true
  })

  if (!particles.length) return null
  return <instancedMesh key={particles.length} ref={mesh} args={[undefined, undefined, particles.length]} frustumCulled={false}>
    <sphereGeometry args={[0.055, 8, 8]} />
    <meshBasicMaterial transparent opacity={0.95} depthWrite={false} toneMapped={false} />
  </instancedMesh>
}

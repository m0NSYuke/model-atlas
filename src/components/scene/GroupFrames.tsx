import { useMemo } from 'react'
import { Html, Line } from '@react-three/drei'
import { Vector3 } from 'three'
import type { ViewFrame } from '../../core/layout/viewGraph'
import { localizedNode } from '../../i18n/translations'
import { useModelStore } from '../../store/modelStore'
import { formatCount, usePalette } from './theme'

function roundedRect(min: [number, number], max: [number, number], z: number, radius = 0.28): Vector3[] {
  const r = Math.min(radius, (max[0] - min[0]) / 2, (max[1] - min[1]) / 2)
  const points: Vector3[] = []
  const corners: [number, number, number][] = [[max[0] - r, max[1] - r, 0], [min[0] + r, max[1] - r, Math.PI / 2], [min[0] + r, min[1] + r, Math.PI], [max[0] - r, min[1] + r, Math.PI * 1.5]]
  for (const [cx, cy, start] of corners) {
    for (let step = 0; step <= 6; step++) {
      const angle = start + (step / 6) * (Math.PI / 2)
      points.push(new Vector3(cx + Math.cos(angle) * r, cy + Math.sin(angle) * r, z))
    }
  }
  points.push(points[0].clone())
  return points
}

/** Translucent panels behind expanded modules, so nesting (block → attention → linear) stays readable. */
function GroupFrame({ frame, maxDepth, minDepth, muted }: { frame: ViewFrame; maxDepth: number; minDepth: number; muted: boolean }) {
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  const selected = useModelStore((state) => state.selectedNodeId === frame.id)
  const hovered = useModelStore((state) => state.hoveredNodeId === frame.id)
  const selectNode = useModelStore((state) => state.selectNode)
  const toggleCollapsed = useModelStore((state) => state.toggleCollapsed)
  const palette = usePalette()
  const frameColors = palette.scene.frames
  const light = palette.theme === 'light'
  const color = frame.node.type === 'module' || frame.node.type === 'generic' ? frameColors[frame.depth % frameColors.length] : palette.types[frame.node.type] ?? '#94a3b8'
  const z = frame.zMin - 0.12 - (maxDepth - frame.depth) * 0.1
  const outline = useMemo(() => roundedRect(frame.min, frame.max, z), [frame.min, frame.max, z])
  const width = frame.max[0] - frame.min[0]
  const height = frame.max[1] - frame.min[1]
  const active = selected || hovered
  const name = localizedNode(frame.node, locale, modelId).name
  return <group>
    <mesh position={[(frame.min[0] + frame.max[0]) / 2, (frame.min[1] + frame.max[1]) / 2, z - 0.01]} onClick={(event) => { event.stopPropagation(); selectNode(frame.id) }}>
      <planeGeometry args={[width, height]} />
      <meshBasicMaterial color={color} transparent opacity={(active ? 0.12 : (light ? 0.07 : 0.05) + frame.depth * 0.012) * (muted ? 0.5 : 1)} depthWrite={false} />
    </mesh>
    <Line points={outline} color={color} lineWidth={active ? 1.8 : 1} transparent opacity={(active ? 0.95 : 0.6) * (muted ? 0.5 : 1)} dashed={frame.depth > 1} dashSize={0.25} gapSize={0.12} depthWrite={false} />
    {frame.segment === 0 && (frame.depth <= minDepth + 1 || active) && <Html position={[frame.min[0] + 0.1, frame.max[1] - 0.05, z]} zIndexRange={[40, 20]} style={{ transform: 'translateY(-100%)' }}>
      <button type="button" className={`frame-tag ${active ? 'active' : ''} ${muted ? 'muted' : ''}`} style={{ ['--accent' as string]: color }}
        title={locale === 'zh' ? '单击选中，双击收起' : 'Click to select, double-click to collapse'}
        onClick={(event) => { event.stopPropagation(); selectNode(frame.id) }}
        onDoubleClick={(event) => { event.stopPropagation(); toggleCollapsed(frame.id, true) }}
        onPointerEnter={() => useModelStore.getState().hoverNode(frame.id)}
        onPointerLeave={() => useModelStore.getState().hoverNode(null)}>
        <i>−</i>{name}{active && frame.node.className && frame.node.className !== frame.node.name && <small>{frame.node.className}</small>}{frame.node.paramCount ? <small>{formatCount(frame.node.paramCount)}</small> : null}
      </button>
    </Html>}
  </group>
}

export function GroupFrames({ frames, focus }: { frames: ViewFrame[]; focus: Set<string> | null }) {
  const maxDepth = Math.max(0, ...frames.map((frame) => frame.depth))
  const minDepth = Math.min(maxDepth, ...frames.map((frame) => frame.depth))
  return <>{frames.map((frame) => <GroupFrame key={`${frame.id}:${frame.segment}`} frame={frame} maxDepth={maxDepth} minDepth={minDepth} muted={Boolean(focus)} />)}</>
}

import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Edges, Html, RoundedBox } from '@react-three/drei'
import { Color, type Group } from 'three'
import type { ViewNode } from '../../core/layout/viewGraph'
import { localizedNode, typeName } from '../../i18n/translations'
import { useModelStore } from '../../store/modelStore'
import { opSymbol, usePalette } from './theme'
import type { DiffMark } from '../../core/study/diff'
import { useStudyStore } from '../../store/studyStore'

export type LabelMode = 'smart' | 'all' | 'none'

type Variant = 'io' | 'param' | 'op' | 'thin' | 'slab' | 'group' | 'card'

function variantOf(item: ViewNode): Variant {
  const { node } = item
  if (item.collapsed) return 'group'
  if (item.card) return node.type === 'tensor' || node.type === 'latent' ? 'io' : 'card'
  if (node.type === 'parameter') return 'param'
  if (node.type === 'tensor' || node.type === 'latent') return 'io'
  if (['activation', 'dropout', 'norm'].includes(node.type)) return 'thin'
  if (['add', 'multiply', 'matmul', 'concat', 'split', 'generic', 'loss', 'reshape', 'optimizer'].includes(node.type) && !node.paramCount) return 'op'
  return 'slab'
}

const ease = (value: number) => 1 - Math.pow(1 - Math.min(1, Math.max(0, value)), 3)

export function LayerGlyph({ item, muted, labels, highlighted, appearStamp, mark, note }: {
  item: ViewNode
  muted: boolean
  labels: LabelMode
  highlighted: boolean
  appearStamp: number
  /** Changed by the current code edits (change comparison). */
  mark?: DiffMark | null
  /** The reader left a note on this block. */
  note?: boolean
}) {
  const { node, size } = item
  const selected = useModelStore((state) => state.selectedNodeId === node.id)
  const hovered = useModelStore((state) => state.hoveredNodeId === node.id)
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  const changedStamp = useModelStore((state) => state.changed.ids.includes(node.id) ? state.changed.stamp : 0)
  const selectNode = useModelStore((state) => state.selectNode)
  const hoverNode = useModelStore((state) => state.hoverNode)
  const toggleCollapsed = useModelStore((state) => state.toggleCollapsed)
  const group = useRef<Group>(null)
  const variant = variantOf(item)
  const palette = usePalette()
  const scene = palette.scene
  const color = palette.types[node.type] ?? '#94a3b8'
  const light = palette.theme === 'light'
  const name = localizedNode(node, locale, modelId).name
  const active = selected || hovered
  const opacity = muted ? 0.2 : 1
  const emissive = useMemo(() => new Color(color), [color])
  const glow = (selected ? 0.75 : hovered ? 0.5 : highlighted ? 0.32 : 0.14) * scene.glow

  useFrame(() => {
    const element = group.current
    if (!element) return
    const now = Date.now()
    const appear = ease((now - appearStamp - item.rank * 22) / 520)
    const pulseAge = changedStamp ? (now - changedStamp) / 1000 : 9
    const pulse = pulseAge < 2.2 ? 1 + Math.sin(pulseAge * 14) * 0.12 * (1 - pulseAge / 2.2) : 1
    const scale = Math.max(0.001, appear * pulse)
    if (element.scale.x !== scale) element.scale.setScalar(scale)
  })

  const [w, h, d] = size
  const plates = useMemo(() => {
    const shape = node.outputShape ?? node.inputShape ?? []
    const channels = typeof shape[1] === 'number' ? shape[1] : 8
    return shape.length >= 4 ? Math.max(2, Math.min(7, Math.round(Math.log2(Math.max(2, channels))) - 1)) : 1
  }, [node.outputShape, node.inputShape])
  const stripes = useMemo(() => {
    const shape = node.outputShape ?? []
    if (shape.length !== 3 || typeof shape[1] !== 'number') return 0
    return Math.max(2, Math.min(12, shape[1]))
  }, [node.outputShape])

  const showLabel = variant === 'card' ? false : labels === 'all' ? variant !== 'op' || active : labels === 'smart' ? active || variant === 'group' || variant === 'io' || node.className?.endsWith('.step') || highlighted && variant !== 'op' : selected
  const shape = node.outputShape ? node.outputShape.join('×') : ''
  const symbol = opSymbol[node.className ?? node.name] ?? opSymbol[node.name] ?? (node.type === 'loss' ? 'ℒ' : '•')

  const material = (fill: number, extra?: { roughness?: number }) => (
    <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={glow} transparent opacity={fill * opacity} roughness={extra?.roughness ?? (light ? 0.55 : 0.42)} metalness={light ? 0.02 : 0.12} depthWrite={fill * opacity > 0.6} />
  )

  return (
    <group
      ref={group}
      position={item.position}
      userData={{ nodeId: node.id }}
      onClick={(event) => {
        event.stopPropagation()
        // Shift+click traces the path from the selected block to this one.
        if (event.nativeEvent.shiftKey && useModelStore.getState().selectedNodeId) useStudyStore.getState().setPathTarget(node.id)
        else selectNode(node.id)
      }}
      onDoubleClick={(event) => { event.stopPropagation(); if (item.collapsed) toggleCollapsed(node.id, false) }}
      onPointerOver={(event) => { event.stopPropagation(); hoverNode(node.id); document.body.style.cursor = 'pointer' }}
      onPointerOut={() => { hoverNode(null); document.body.style.cursor = 'auto' }}
    >
      {/* Generous invisible hit box so thin plates stay easy to click. */}
      <mesh><boxGeometry args={[Math.max(w, 0.5) + 0.2, h + 0.2, d + 0.2]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} /></mesh>

      {variant === 'slab' && plates > 1 && Array.from({ length: plates }, (_, index) => {
        const x = plates === 1 ? 0 : -w / 2 + (index / (plates - 1)) * w
        return <mesh key={index} position={[x, 0, 0]}><boxGeometry args={[Math.min(0.05, w / plates), h, d]} />{material(index === plates - 1 ? 0.9 : 0.42)}</mesh>
      })}
      {variant === 'slab' && plates > 1 && <mesh><boxGeometry args={[w, h, d]} /><meshBasicMaterial color={color} transparent opacity={0.05 * opacity} depthWrite={false} /><Edges color={color} transparent opacity={(active ? 1 : 0.55) * opacity} /></mesh>}
      {variant === 'slab' && plates <= 1 && <RoundedBox args={[w, h, d]} radius={Math.min(0.08, w / 3, h / 3, d / 3)} smoothness={2}>{material(light ? 0.94 : 0.86)}<Edges color={active ? scene.outline : color} transparent opacity={(active ? 0.9 : light ? 0.25 : 0.4) * opacity} threshold={25} /></RoundedBox>}
      {variant === 'slab' && stripes > 0 && Array.from({ length: stripes - 1 }, (_, index) => (
        <mesh key={`s${index}`} position={[w / 2 + 0.006, -h / 2 + ((index + 1) / stripes) * h, 0]}><boxGeometry args={[0.012, 0.012, d * 0.96]} /><meshBasicMaterial color={scene.stripe} transparent opacity={(light ? 0.8 : 0.55) * opacity} /></mesh>
      ))}

      {variant === 'thin' && <mesh><boxGeometry args={[w, h, d]} />{material(light ? 0.6 : 0.42, { roughness: 0.2 })}<Edges color={color} transparent opacity={(active ? 1 : 0.6) * opacity} /></mesh>}

      {variant === 'io' && <>
        <mesh><boxGeometry args={[w, h, d]} /><meshBasicMaterial color={color} transparent opacity={(active ? 0.24 : light ? 0.12 : 0.08) * opacity} depthWrite={false} /><Edges color={color} transparent opacity={opacity} /></mesh>
        {Array.from({ length: Math.min(4, plates + 1) }, (_, index) => (
          <mesh key={index} position={[-w / 2 + ((index + 1) / (Math.min(4, plates + 1) + 1)) * w, 0, 0]}><boxGeometry args={[0.01, h * 0.92, d * 0.92]} /><meshBasicMaterial color={color} transparent opacity={0.22 * opacity} depthWrite={false} /></mesh>
        ))}
      </>}

      {variant === 'param' && <mesh scale={[w, h * 0.7, d * 0.7]}><octahedronGeometry args={[0.72, 0]} />{material(0.7)}<Edges color={color} transparent opacity={0.8 * opacity} /></mesh>}

      {variant === 'op' && <mesh><icosahedronGeometry args={[0.27, 1]} /><meshStandardMaterial color={scene.op} emissive={emissive} emissiveIntensity={(selected ? 0.9 : hovered ? 0.6 : 0.25) * scene.glow} roughness={0.25} metalness={0.2} transparent opacity={opacity} /></mesh>}

      {variant === 'group' && <RoundedBox args={[w, h, d]} radius={0.14} smoothness={3}>
        <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={(selected ? 0.6 : hovered ? 0.45 : 0.28) * scene.glow} transparent opacity={(light ? 0.78 : 0.62) * opacity} roughness={0.25} metalness={0.1} />
        <Edges color={color} transparent opacity={opacity} threshold={20} />
      </RoundedBox>}

      {variant === 'card' && <RoundedBox args={[w, h, d]} radius={0.1} smoothness={3}>
        <meshStandardMaterial color={active ? scene.cardActive : scene.card} emissive={emissive} emissiveIntensity={(selected ? 0.22 : hovered ? 0.14 : 0.05) * scene.glow} transparent opacity={opacity} roughness={0.5} metalness={0.2} />
        <Edges color={color} transparent opacity={(active ? 1 : 0.55) * opacity} threshold={20} />
      </RoundedBox>}
      {variant === 'card' && <mesh position={[-w / 2 + 0.05, 0, d / 2 + 0.005]}><boxGeometry args={[0.05, h * 0.7, 0.01]} /><meshBasicMaterial color={color} transparent opacity={opacity} /></mesh>}
      {variant === 'card' && <Html center distanceFactor={15} position={[0.04, 0, d / 2 + 0.02]} zIndexRange={[8, 0]} style={{ pointerEvents: 'none' }}>
        <div className={`card-label ${muted ? 'muted' : ''} ${selected ? 'selected' : ''}`} style={{ ['--accent' as string]: color }}>
          <span>{typeName(locale, node.type)}</span>
          <strong>{name}</strong>
          {labels !== 'none' && shape && <em>{shape}</em>}
        </div>
      </Html>}

      {selected && <mesh><boxGeometry args={[Math.max(w, 0.56) + 0.24, h + 0.24, d + 0.24]} /><meshBasicMaterial color={color} transparent opacity={light ? 0.12 : 0.1} depthWrite={false} /><Edges color={scene.outline} transparent opacity={0.85} /></mesh>}

      {(mark || note) && !muted && <Html position={[-Math.max(w, 0.5) / 2, h / 2, d / 2]} center zIndexRange={[12, 0]} style={{ pointerEvents: 'none' }}>
        <span className="glyph-badges">{mark && <span className={`diff-badge ${mark}`}>{mark === 'added' ? '+' : 'Δ'}</span>}{note && <span className="note-badge">✎</span>}</span>
      </Html>}
      {variant === 'op' && !muted && <Html center position={[0, 0, 0.32]} zIndexRange={[5, 0]} style={{ pointerEvents: 'none' }}><span className="glyph-symbol">{symbol}</span></Html>}
      {showLabel && !muted && <Html center position={[0, h / 2 + (variant === 'op' ? 0.45 : 0.42), 0]} zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
        <div className={`glyph-label ${selected ? 'selected' : ''} ${variant === 'group' ? 'group' : ''}`} style={{ ['--accent' as string]: color }}>
          <strong>{name}</strong>
          {(active || labels === 'all') && shape && <span>{shape}</span>}
        </div>
      </Html>}
    </group>
  )
}

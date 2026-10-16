import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Grid } from '@react-three/drei'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { AlertTriangle, Box, Camera, ShieldCheck, ChevronDown, Focus, Keyboard, Layers, Maximize2, Minimize2, Minus, Plus, Rotate3d, Route, Settings2, Sparkles, Tag, TagsIcon, Waypoints, X } from 'lucide-react'
import { effectiveDepth, useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { localizedModel, localizedNode, t, typeName } from '../../i18n/translations'
import { codeModelById } from '../../data/models/codeCatalog'
import { useHandTracking } from '../../gestures/useHandTracking'
import { GestureSceneInput } from '../three/GestureSceneInput'
import type { ModelIR, ModelNodeType } from '../../core/model-ir/types'
import { LayerGlyph, type LabelMode } from './LayerGlyph'
import { FlowEdges, FlowParticles, useEdgePaths } from './FlowEdges'
import { GroupFrames } from './GroupFrames'
import { CameraRig, type CameraAngle } from './CameraRig'
import { DiagramView } from './DiagramView'
import { tr, useViewGraph } from './useViewGraph'
import { formatCount, legendTypes, usePalette } from './theme'
import type { ViewGraph } from '../../core/layout/viewGraph'
import { notesKey, useDiffMarks, useStudyStore } from '../../store/studyStore'
import { tracePath } from '../../core/study/path'
import { checkScene } from './drawingCheck'

/** Groups node types into the legend buckets shown under the scene. */
const legendBucket: Partial<Record<ModelNodeType, ModelNodeType>> = {
  conv1d: 'conv2d', conv3d: 'conv2d', downsample: 'conv2d', mlp: 'linear', multiply: 'add', matmul: 'add', concat: 'add', split: 'add', reshape: 'add', generic: 'add', dropout: 'activation', latent: 'tensor', sampling: 'tensor', encoder: 'module', decoder: 'module',
}

function focusSet(graph: ViewGraph, id: string | null): Set<string> | null {
  if (!id) return null
  const visible = graph.representative.get(id) ?? id
  const set = new Set([visible])
  const frame = graph.frames.some((item) => item.id === id)
  if (frame) {
    // A selected expanded group focuses on all of its visible members.
    const model = useModelStore.getState().modelIR
    const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
    graph.nodes.forEach((item) => { for (let current = item.node.parentId; current; current = parents.get(current)) if (current === id) set.add(item.id) })
  }
  const members = new Set(set)
  graph.edges.forEach((edge) => {
    if (members.has(edge.source)) set.add(edge.target)
    if (members.has(edge.target)) set.add(edge.source)
  })
  return set
}

/** The main path from the first input to the output, reduced to the points where the tensor shape changes. */
function tensorJourney(model: ModelIR): { id: string; name: string; shape: string }[] {
  if (model.layout !== 'code') return []
  const groups = new Set(model.nodes.map((node) => node.parentId).filter(Boolean))
  const leaves = model.nodes.filter((node) => !groups.has(node.id) && node.type !== 'parameter')
  const leafIds = new Set(leaves.map((node) => node.id))
  const successors = new Map<string, string[]>()
  model.edges.forEach((edge) => { if (leafIds.has(edge.source) && leafIds.has(edge.target) && edge.type !== 'condition') successors.set(edge.source, [...(successors.get(edge.source) ?? []), edge.target]) })
  const start = leaves.find((node) => node.className === 'Input')
  if (!start) return []
  // Longest path in trace order (the trace is topologically sorted).
  const distance = new Map<string, number>([[start.id, 0]])
  const previous = new Map<string, string>()
  for (const node of leaves) {
    const base = distance.get(node.id)
    if (base === undefined) continue
    for (const next of successors.get(node.id) ?? []) {
      if ((distance.get(next) ?? -1) < base + 1) { distance.set(next, base + 1); previous.set(next, node.id) }
    }
  }
  const output = leaves.filter((node) => node.className === 'Output' && distance.has(node.id)).sort((a, b) => distance.get(b.id)! - distance.get(a.id)!)[0]
  const farthest = [...distance.entries()].sort((a, b) => b[1] - a[1])[0]?.[0]
  const end = output ?? leaves.find((node) => node.id === farthest)
  if (!end) return []
  const path: string[] = []
  for (let current: string | undefined = end.id; current; current = previous.get(current)) path.unshift(current)
  const byId = new Map(model.nodes.map((node) => [node.id, node]))
  const steps: { id: string; name: string; shape: string }[] = []
  for (const id of path) {
    const node = byId.get(id)!
    const shape = node.outputShape?.join('×') ?? ''
    if (!shape || steps.at(-1)?.shape === shape) continue
    steps.push({ id, name: node.name, shape })
  }
  return steps
}

function SceneContents({ graph, path, modelId, labels, trace, flow, fitRequest, angle, controls, hand, mousePauseUntil, onInspect }: {
  graph: ViewGraph; path: string[] | null; modelId: string; labels: LabelMode; trace: boolean; flow: boolean; fitRequest: number; angle: CameraAngle
  controls: React.RefObject<OrbitControlsImpl>; hand: ReturnType<typeof useHandTracking>; mousePauseUntil: React.RefObject<number>; onInspect: (id: string) => void
}) {
  const selected = useModelStore((state) => state.selectedNodeId)
  const hovered = useModelStore((state) => state.hoveredNodeId)
  const appearStamp = useModelStore((state) => state.changed.stamp)
  const { scene } = usePalette()
  const marks = useDiffMarks()
  const key = useModelStore(notesKey)
  const notes = useStudyStore((state) => state.notes[key])
  const focus = useMemo(() => path ? new Set(path) : focusSet(graph, hovered ?? selected), [graph, hovered, selected, path])
  const paths = useEdgePaths(graph)
  const { min, max } = graph.bounds
  return <>
    <ambientLight intensity={scene.ambient} />
    <hemisphereLight args={scene.hemisphere} />
    <directionalLight position={[-12, 18, 22]} intensity={scene.key} />
    <directionalLight position={[16, -6, 10]} intensity={0.55} color={scene.rim} />
    <directionalLight position={[4, 10, -16]} intensity={0.35} />
    <fog attach="fog" args={[scene.fog, 70, 240]} />
    <GroupFrames frames={graph.frames} focus={trace ? focus : null} />
    <FlowEdges paths={paths} focus={focus} trace={trace || Boolean(path)} path={path} />
    {flow && <FlowParticles paths={paths} focus={trace ? focus : null} />}
    {graph.nodes.map((item) => <LayerGlyph key={item.id} item={item} labels={labels} appearStamp={appearStamp}
      muted={Boolean((trace || path) && focus && !focus.has(item.id))} highlighted={Boolean(focus?.has(item.id))}
      mark={marks?.get(item.id)} note={Boolean(notes?.[item.id])} />)}
    <Grid position={[(min[0] + max[0]) / 2, min[1] - 2.2, 0]} args={[10, 10]} infiniteGrid cellSize={1} sectionSize={5} cellThickness={0.5} sectionThickness={0.9}
      cellColor={scene.grid.cell} sectionColor={scene.grid.section} fadeDistance={Math.max(60, (max[0] - min[0]) * 1.4)} fadeStrength={1.6} followCamera={false} />
    <CameraRig graph={graph} modelId={modelId} fitRequest={fitRequest} angle={angle} controls={controls} />
    {hand.status === 'tracking' && <GestureSceneInput intentRef={hand.intentRef} controls={controls} sensitivity={hand.sensitivity} mousePauseUntil={mousePauseUntil} onInspect={onInspect} />}
  </>
}

function HoverCard({ graph }: { graph: ViewGraph }) {
  const hovered = useModelStore((state) => state.hoveredNodeId)
  const locale = useModelStore((state) => state.locale)
  const model = useModelStore((state) => state.modelIR)
  const { types } = usePalette()
  const [point, setPoint] = useState<{ x: number; y: number } | null>(null)
  useEffect(() => {
    const move = (event: PointerEvent) => setPoint({ x: event.clientX, y: event.clientY })
    window.addEventListener('pointermove', move)
    return () => window.removeEventListener('pointermove', move)
  }, [])
  const item = hovered ? graph.byId.get(hovered) ?? null : null
  const node = item?.node ?? (hovered ? model.nodes.find((entry) => entry.id === hovered) : undefined)
  if (!node || !point) return null
  const name = localizedNode(node, locale, model.id).name
  const flip = point.x > window.innerWidth - 300
  return <div className="hover-card" style={{ left: flip ? point.x - 16 : point.x + 16, top: point.y + 14, transform: flip ? 'translateX(-100%)' : undefined, ['--accent' as string]: types[node.type] }}>
    <div className="hover-card-head"><i />{typeName(locale, node.type)}{node.className && <code>{node.className}</code>}</div>
    <strong>{name}</strong>
    {(node.inputShape || node.outputShape) && <div className="hover-card-shapes">
      {node.inputShape && <span>{node.inputShape.join('×')}</span>}
      {node.inputShape && node.outputShape && <b>→</b>}
      {node.outputShape && <span>{node.outputShape.join('×')}</span>}
    </div>}
    {node.paramCount ? <small>{formatCount(node.paramCount)} {tr(locale, 'params', '参数')}</small> : null}
    {item?.collapsed && <small className="hint">{tr(locale, 'Double-click to expand', '双击展开内部结构')}</small>}
  </div>
}

/** Live check of what is drawn: every port connected, shapes agree, no line through a block. */
function IntegrityChip({ graph }: { graph: ViewGraph }) {
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const report = useMemo(() => checkScene(model, graph), [model, graph])
  const errors = report.ports.length + report.shapes.length + report.overlaps.length + report.through.length
  const detail = [
    tr(locale, `Ports ${report.ports.length}`, `接口错误 ${report.ports.length}`),
    tr(locale, `Shape mismatches ${report.shapes.length}`, `形状不符 ${report.shapes.length}`),
    tr(locale, `Overlaps ${report.overlaps.length}`, `方块重叠 ${report.overlaps.length}`),
    tr(locale, `Lines through blocks ${report.through.length}`, `连线穿过方块 ${report.through.length}`),
    tr(locale, `Crossings ${report.crossings.length}`, `连线交叉 ${report.crossings.length}`),
    ...[...report.ports, ...report.shapes, ...report.through].slice(0, 4),
  ].join('\n')
  return <span className={`integrity-chip ${errors ? 'warn' : 'ok'}`} title={detail}>
    {errors ? <AlertTriangle size={12} /> : <ShieldCheck size={12} />}
    {errors ? tr(locale, `${errors} wiring issues`, `${errors} 处连线问题`) : tr(locale, 'Wiring verified', '连线已校验')}
    {!errors && report.crossings.length > 0 && <small>{tr(locale, `${report.crossings.length} crossings`, `${report.crossings.length} 处交叉`)}</small>}
  </span>
}

/** Model name, family and live statistics, floating over the top-left of the stage. */
function StageHeader({ graph }: { graph: ViewGraph }) {
  const model = useModelStore((state) => state.modelIR)
  const origin = useModelStore((state) => state.modelOrigin)
  const builtInId = useModelStore((state) => state.builtInId)
  const locale = useModelStore((state) => state.locale)
  const diagnostic = useModelStore((state) => state.diagnostic)
  const setLibraryOpen = useModelStore((state) => state.setLibraryOpen)
  const entry = builtInId ? codeModelById[builtInId] : undefined
  const stats = useMemo(() => {
    const groups = new Set(model.nodes.map((node) => node.parentId).filter(Boolean))
    const layers = model.nodes.filter((node) => !groups.has(node.id) && node.paramCount).length
    return { layers, nodes: model.nodes.length, edges: model.edges.length }
  }, [model])
  const badge = origin === 'code' ? tr(locale, 'Live PyTorch', 'PyTorch 实时') : origin === 'import' ? tr(locale, 'Imported', '已导入') : origin === 'demo' ? tr(locale, 'Editable demo', '可编辑示例') : tr(locale, 'Study map', '学习图谱')
  return <div className="stage-header">
    <button type="button" className="stage-title" onClick={() => setLibraryOpen(true)} title={tr(locale, 'Switch model (Ctrl+K)', '切换模型（Ctrl+K）')}>
      <span className="stage-title-name">{localizedModel(model, locale)}</span>
      <ChevronDown size={15} />
    </button>
    <div className="stage-meta">
      {entry && <span className="meta-chip">{entry.family[locale]}</span>}
      <span className={`meta-chip origin ${origin}`}>{badge}</span>
      {origin === 'code' && <span className={`sync ${diagnostic ? 'error' : 'ok'}`}><i />{diagnostic ? tr(locale, 'Code error', '代码有误') : tr(locale, 'In sync', '已同步')}</span>}
      <IntegrityChip graph={graph} />
    </div>
    <div className="stage-stats">
      {model.totalParams !== undefined && <div><b>{formatCount(model.totalParams)}</b><span>{tr(locale, 'params', '参数')}</span></div>}
      {origin === 'code' && <div><b>{stats.layers}</b><span>{tr(locale, 'weighted layers', '含权重层')}</span></div>}
      <div><b>{stats.nodes}</b><span>{tr(locale, 'nodes', '节点')}</span></div>
      <div><b>{stats.edges}</b><span>{tr(locale, 'edges', '连线')}</span></div>
    </div>
  </div>
}

function Legend({ graph }: { graph: ViewGraph }) {
  const locale = useModelStore((state) => state.locale)
  const { types, edges } = usePalette()
  const [open, setOpen] = useState(false)
  const presentTypes = useMemo(() => {
    const present = new Set(graph.nodes.map((item) => legendBucket[item.node.type] ?? item.node.type))
    return legendTypes.filter((entry) => present.has(entry.type))
  }, [graph])
  const edgeTypes = useMemo(() => [...new Set(graph.edges.map((edge) => edge.type ?? 'data'))], [graph])
  const edgeNames: Record<string, [string, string]> = { data: ['Data flow', '数据流'], residual: ['Residual', '残差'], branch: ['Skip', '跳连'], condition: ['Condition', '条件'], training: ['Training', '训练'], feedback: ['Feedback', '回路'], reuse: ['Shared', '复用'] }
  if (!presentTypes.length) return null
  return <div className={`legend ${open ? 'open' : ''}`}>
    {open && <div className="legend-panel glass" role="dialog" aria-label={tr(locale, 'Legend', '图例')}>
      <div className="legend-title">{tr(locale, 'Layer types', '层类型')}</div>
      <div className="legend-grid">{presentTypes.map((entry) => <span key={entry.type}><i style={{ background: types[entry.type] }} />{locale === 'zh' ? entry.zh : entry.en}</span>)}</div>
      {edgeTypes.length > 0 && <>
        <div className="legend-title">{tr(locale, 'Connections', '连线')}</div>
        <div className="legend-grid">{edgeTypes.map((type) => <span key={type}><em style={{ background: edges[type] }} />{tr(locale, ...(edgeNames[type] ?? [type, type]))}</span>)}</div>
      </>}
    </div>}
    <button type="button" className="legend-button glass" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
      <span className="legend-dots">{presentTypes.slice(0, 5).map((entry) => <i key={entry.type} style={{ background: types[entry.type] }} />)}</span>
      {tr(locale, 'Legend', '图例')}
    </button>
  </div>
}

function ToolButton({ tip, pressed, onClick, children, disabled }: { tip: string; pressed?: boolean; onClick: () => void; children: React.ReactNode; disabled?: boolean }) {
  return <button type="button" className="tool" data-tip={tip} data-tip-side="left" aria-label={tip} aria-pressed={pressed} onClick={onClick} disabled={disabled}>{children}</button>
}

export function ModelScene() {
  const hand = useHandTracking()
  const controls = useRef<OrbitControlsImpl>(null)
  const mousePauseUntil = useRef(0)
  const graph = useViewGraph()
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const diagnostic = useModelStore((state) => state.diagnostic)
  const changed = useModelStore((state) => state.changed)
  const depthSetting = useModelStore((state) => state.depth)
  const depth = useModelStore((state) => effectiveDepth(state))
  const setDepth = useModelStore((state) => state.setDepth)
  const selectNode = useModelStore((state) => state.selectNode)
  const openSource = useModelStore((state) => state.openSource)
  const selectedId = useModelStore((state) => state.selectedNodeId)
  const view = useUiStore((state) => state.view)
  const angle = useUiStore((state) => state.angle)
  const labels = useUiStore((state) => state.labels)
  const trace = useUiStore((state) => state.trace)
  const flow = useUiStore((state) => state.flow)
  const fitRequest = useUiStore((state) => state.fitRequest)
  const focusMode = useUiStore((state) => state.focusMemory !== null || (!state.codeOpen && !state.sideOpen))
  const ui = useUiStore.getState
  const [gestureSettings, setGestureSettings] = useState(false)
  const [toast, setToast] = useState<string | null>(null)
  const isCode = model.layout === 'code'
  const journey = useMemo(() => tensorJourney(model), [model])
  const pathTarget = useStudyStore((state) => state.pathTarget)
  const [pathHint, setPathHint] = useState(false)
  useEffect(() => {
    if (!selectedId) return
    setPathHint(true)
    const timer = setTimeout(() => setPathHint(false), 3200)
    return () => clearTimeout(timer)
  }, [selectedId])
  const path = useMemo(() => selectedId && pathTarget ? tracePath(graph, selectedId, pathTarget) : null, [graph, selectedId, pathTarget])

  useEffect(() => { ui().fit() }, [depthSetting, ui])
  // Re-frame the model once a dock finishes opening or closing, unless the user is looking at a selection.
  const codeOpen = useUiStore((state) => state.codeOpen)
  const sideOpen = useUiStore((state) => state.sideOpen)
  const firstLayout = useRef(true)
  useEffect(() => {
    if (firstLayout.current) { firstLayout.current = false; return }
    const timer = setTimeout(() => { if (!useModelStore.getState().selectedNodeId) ui().fit() }, 420)
    return () => clearTimeout(timer)
  }, [codeOpen, sideOpen, ui])
  useEffect(() => {
    if (!changed.ids.length) return
    setToast(tr(locale, `${changed.ids.length} node${changed.ids.length > 1 ? 's' : ''} changed`, `${changed.ids.length} 个节点随代码更新`))
    const timer = setTimeout(() => setToast(null), 2600)
    return () => clearTimeout(timer)
  }, [changed, locale])
  useEffect(() => {
    const release = () => { if (mousePauseUntil.current === Infinity) mousePauseUntil.current = performance.now() + 1000 }
    window.addEventListener('pointerup', release)
    return () => window.removeEventListener('pointerup', release)
  }, [])

  const inspect = (id: string) => { selectNode(id); ui().setTrace(true) }
  const handOn = hand.status !== 'off' && hand.status !== 'error'
  const labelText = { smart: tr(locale, 'Smart labels', '智能标签'), all: tr(locale, 'All labels', '全部标签'), none: tr(locale, 'No labels', '隐藏标签') }

  return <section className={`stage ${view === '2d' ? 'stage-2d' : ''}`}>
    <div className="stage-backdrop" aria-hidden="true" />
    <div className="canvas-wrap" onPointerDownCapture={() => { mousePauseUntil.current = Infinity }} onWheelCapture={() => { mousePauseUntil.current = performance.now() + 1000 }}
      onPointerLeave={() => useModelStore.getState().hoverNode(null)}>
      {view === '2d' ? <DiagramView graph={graph} trace={trace || Boolean(path)} labels={labels} fitRequest={fitRequest} focus={path ? new Set(path) : focusSet(graph, selectedId)} path={path} />
        : <Canvas camera={{ position: [0, 6, 30], fov: 40, near: 0.1, far: 600 }} dpr={[1, 2]} gl={{ antialias: true, alpha: true }} onPointerMissed={() => useModelStore.getState().resetSelection()}>
          <SceneContents graph={graph} path={path} modelId={model.id} labels={labels} trace={trace} flow={flow} fitRequest={fitRequest} angle={angle} controls={controls} hand={hand} mousePauseUntil={mousePauseUntil} onInspect={inspect} />
        </Canvas>}
    </div>

    <StageHeader graph={graph} />

    <div className="stage-toolbar glass">
      <div className="segmented" role="group" aria-label={tr(locale, 'View', '视图')}>
        <button type="button" aria-pressed={view === '3d'} onClick={() => ui().setView('3d')}><Box size={14} /><span className="seg-label">3D</span></button>
        <button type="button" aria-pressed={view === '2d'} onClick={() => { hand.stop(); ui().setView('2d') }}><Waypoints size={14} /><span className="seg-label">{tr(locale, 'Diagram', '结构图')}</span></button>
      </div>
      {isCode && <div className="depth-control" data-tip={tr(locale, 'Levels of nested modules shown', '展开的嵌套层级')}>
        <Layers size={14} />
        <button type="button" disabled={depth <= 0} onClick={() => setDepth(Math.max(0, depth - 1))} aria-label={tr(locale, 'Collapse one level', '收起一层')}><Minus size={12} /></button>
        <b>{depth >= graph.maxDepth ? tr(locale, 'All', '全部') : depth}</b>
        <button type="button" disabled={depth >= graph.maxDepth} onClick={() => setDepth(Math.min(graph.maxDepth, depth + 1))} aria-label={tr(locale, 'Expand one level', '展开一层')}><Plus size={12} /></button>
        <button type="button" className={`depth-auto ${depthSetting === null ? 'on' : ''}`} onClick={() => setDepth(null)}>{tr(locale, 'Auto', '自动')}</button>
      </div>}
    </div>

    <div className="stage-tools glass" role="toolbar" aria-label={tr(locale, 'View tools', '视图工具')}>
      <ToolButton tip={`${labelText[labels]} (L)`} pressed={labels !== 'none'} onClick={() => ui().cycleLabels()}>{labels === 'all' ? <TagsIcon size={15} /> : <Tag size={15} />}</ToolButton>
      <ToolButton tip={t(locale, 'traceHint')} pressed={trace} onClick={() => ui().setTrace(!trace)}><Route size={15} /></ToolButton>
      {view === '3d' && <ToolButton tip={tr(locale, 'Tensor flow animation', '张量流动动画')} pressed={flow} onClick={() => ui().setFlow(!flow)}><Sparkles size={15} /></ToolButton>}
      {view === '3d' && <ToolButton tip={angle === 'iso' ? tr(locale, 'Front view', '切换到正视') : tr(locale, 'Angled view', '切换到斜视')} pressed={angle === 'front'} onClick={() => ui().setAngle(angle === 'iso' ? 'front' : 'iso')}><Rotate3d size={15} /></ToolButton>}
      {view === '3d' && <ToolButton tip={t(locale, handOn ? 'stopGesture' : 'startGesture')} pressed={handOn} onClick={() => { if (handOn) hand.stop(); else { ui().toggleFocusMode(true); void hand.start() } }}><Camera size={15} /></ToolButton>}
      <span className="tools-divider" />
      <ToolButton tip={`${t(locale, 'fit')} (F)`} onClick={() => ui().fit()}><Focus size={15} /></ToolButton>
      <ToolButton tip={focusMode ? tr(locale, 'Show panels (\\)', '显示侧栏（\\）') : tr(locale, 'Focus mode (\\)', '专注模式（\\）')} pressed={focusMode} onClick={() => ui().toggleFocusMode()}>{focusMode ? <Minimize2 size={15} /> : <Maximize2 size={15} />}</ToolButton>
    </div>

    <div className="stage-notices">
      {diagnostic && <div className="error-banner" role="alert">
        <AlertTriangle size={16} />
        <div><strong>{tr(locale, 'The code no longer runs — showing the last working graph', '代码当前无法运行，下方保留的是上一次成功的结构')}</strong><span>{diagnostic.message}</span></div>
        {diagnostic.line && <button type="button" onClick={() => { ui().togglePanel('codeOpen', true); openSource(diagnostic.file ?? Object.keys(useModelStore.getState().codeFiles)[0], diagnostic.line!) }}>{tr(locale, `Line ${diagnostic.line}`, `第 ${diagnostic.line} 行`)}</button>}
      </div>}
      {toast && !diagnostic && <div className="change-toast glass"><Sparkles size={14} />{toast}</div>}
      {path && <div className="path-chip glass">
        <Route size={14} />
        <span>{tr(locale, 'Path', '路径')}</span>
        <b>{localizedNode(graph.byId.get(path[0])!.node, locale, model.id).name}</b><i>→</i><b>{localizedNode(graph.byId.get(path[path.length - 1])!.node, locale, model.id).name}</b>
        <small>{tr(locale, `${path.length - 1} hops`, `${path.length - 1} 步`)}</small>
        <button type="button" onClick={() => useStudyStore.getState().setPathTarget(null)} aria-label={tr(locale, 'Clear path', '清除路径')}><X size={13} /></button>
      </div>}
      {!path && selectedId && pathHint && <div className="path-hint">{tr(locale, 'Shift+click another block to trace the path to it', 'Shift+点击另一个模块，查看两者之间的路径')}</div>}
    </div>

    <div className="stage-footer">
      <Legend graph={graph} />
      {journey.length > 1 && <div className="journey glass" aria-label={tr(locale, 'Tensor shape along the main path', '主路径上的张量形状变化')}>
        <span className="journey-title">{tr(locale, 'Tensor journey', '张量旅程')}</span>
        <div className="journey-track">
          {journey.map((step, index) => <span key={step.id + index} className="journey-step">
            {index > 0 && <i />}
            <button type="button" className={selectedId === step.id ? 'active' : ''} onClick={() => selectNode(step.id)} title={step.name}>{step.shape}</button>
          </span>)}
        </div>
      </div>}
      <button type="button" className="shortcut-hint glass" onClick={() => ui().setOverlay('shortcuts')} title={tr(locale, 'Keyboard shortcuts', '键盘快捷键')}>
        <Keyboard size={13} /><span>{tr(locale, 'Shortcuts', '快捷键')}</span><kbd>?</kbd>
      </button>
    </div>

    <div className={`gesture-monitor ${handOn ? 'active' : ''}`}>
      <video ref={hand.videoRef} className={hand.mirror ? 'mirrored' : ''} muted playsInline autoPlay aria-label={t(locale, 'cameraPreview')} />
      {handOn && <div className="gesture-monitor-info"><span className={`gesture-status-dot ${hand.status}`} />{t(locale, `gestureStatus_${hand.status}`)}<button type="button" title={t(locale, 'gestureSettings')} onClick={() => setGestureSettings((value) => !value)}><Settings2 size={13} /></button></div>}
      {handOn && gestureSettings && <div className="gesture-settings">
        {hand.devices.length > 0 && <label>{t(locale, 'gestureCamera')}<select value={hand.deviceId} onChange={(event) => void hand.start(event.target.value)}>{hand.devices.map((device, index) => <option key={device.deviceId} value={device.deviceId}>{device.label || `${t(locale, 'gestureCamera')} ${index + 1}`}</option>)}</select></label>}
        <label>{t(locale, 'gestureHand')}<select value={hand.preferredHand} onChange={(event) => hand.setPreferredHand(event.target.value as typeof hand.preferredHand)}><option value="any">{t(locale, 'gestureAnyHand')}</option><option value="left">{t(locale, 'gestureLeftHand')}</option><option value="right">{t(locale, 'gestureRightHand')}</option></select></label>
        <label>{t(locale, 'gestureSensitivity')}<input type="range" min="0.5" max="2" step="0.1" value={hand.sensitivity} onChange={(event) => hand.setSensitivity(Number(event.target.value))} /></label>
        <label className="gesture-mirror"><input type="checkbox" checked={hand.mirror} onChange={(event) => hand.setMirror(event.target.checked)} />{t(locale, 'gestureMirror')}</label>
      </div>}
    </div>
    {view === '3d' && hand.uiIntent && <div className={`gesture-cursor ${hand.uiIntent.pinched ? 'pinched' : hand.uiIntent.pointing ? 'pointing' : ''}`} style={{ left: `${hand.uiIntent.x * 100}%`, top: `${hand.uiIntent.y * 100}%` }}><span /></div>}
    {view === '3d' && hand.status === 'error' && <div className="gesture-error"><span>{t(locale, `gestureError_${hand.error ?? 'unknown'}`)}{hand.error === 'model' && hand.errorDetail && <small>{hand.errorDetail}</small>}</span><button type="button" onClick={() => hand.stop()}><X size={14} /></button></div>}
    {view === '3d' && hand.status === 'tracking' && <div className="gesture-hint glass">{t(locale, 'gestureHint')}</div>}
    <HoverCard graph={graph} />
  </section>
}

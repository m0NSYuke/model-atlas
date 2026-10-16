import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fanouts, type ViewGraph } from '../../core/layout/viewGraph'
import { STUB, diagramBoxes, diagramEdgePath, diagramMetrics, samplePath, stubX } from './diagramGeometry'
import { localizedNode, typeName } from '../../i18n/translations'
import { useModelStore } from '../../store/modelStore'
import type { LabelMode } from './LayerGlyph'
import { formatCount, opSymbol, usePalette } from './theme'
import { useViewGraph } from './useViewGraph'
import { notesKey, useDiffMarks, useStudyStore } from '../../store/studyStore'

/** The point halfway along a polyline. */
function midpoint(points: [number, number][]): [number, number] {
  const lengths = points.slice(1).map((point, index) => Math.hypot(point[0] - points[index][0], point[1] - points[index][1]))
  let half = lengths.reduce((sum, value) => sum + value, 0) / 2
  for (let index = 0; index < lengths.length; index++) {
    if (half <= lengths[index]) { const t = half / (lengths[index] || 1); return [points[index][0] + (points[index + 1][0] - points[index][0]) * t, points[index][1] + (points[index + 1][1] - points[index][1]) * t] }
    half -= lengths[index]
  }
  return points[points.length - 1] ?? [0, 0]
}

/**
 * A flat, readable diagram: pan with drag, zoom with the wheel. Code models run the same layout as the
 * 3D scene (U bands, snake rows, routed bridges) with card-sized metrics and a fold that fits this viewport.
 */
export function DiagramView({ graph: sceneGraph, trace, labels, fitRequest, focus, path }: { graph: ViewGraph; trace: boolean; labels: LabelMode; fitRequest: number; focus: Set<string> | null; path?: string[] | null }) {
  const hops = path ? new Set(path.slice(1).flatMap((id, index) => [`${path[index]}>${id}`, `${id}>${path[index]}`])) : null
  const marks = useDiffMarks()
  const key = useModelStore(notesKey)
  const notes = useStudyStore((state) => state.notes[key])
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  const selected = useModelStore((state) => state.selectedNodeId)
  const hovered = useModelStore((state) => state.hoveredNodeId)
  const selectNode = useModelStore((state) => state.selectNode)
  const hoverNode = useModelStore((state) => state.hoverNode)
  const toggleCollapsed = useModelStore((state) => state.toggleCollapsed)
  const { types: typeColor, edges: edgeColor, scene } = usePalette()
  const viewport = useRef<HTMLDivElement>(null)
  const [view, setView] = useState({ x: 0, y: 0, k: 1 })
  const drag = useRef<{ x: number; y: number; vx: number; vy: number; moved: boolean } | null>(null)
  const manual = sceneGraph.kind === 'manual'

  const [aspect, setAspect] = useState(1.6)
  useEffect(() => {
    const element = viewport.current
    if (!element) return
    const observer = new ResizeObserver(([entry]) => setAspect(Math.round(Math.max(0.7, entry.contentRect.width / Math.max(1, entry.contentRect.height - 220)) * 10) / 10))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const metrics = useMemo(() => diagramMetrics(aspect), [aspect])
  const flatGraph = useViewGraph(manual ? null : metrics)
  const graph = manual || !flatGraph ? sceneGraph : flatGraph
  const boxes = useMemo(() => diagramBoxes(graph), [graph])
  const frames = useMemo(() => graph.frames.map((frame) => ({
    frame, segment: frame.segment, x: frame.min[0], y: -frame.max[1], w: frame.max[0] - frame.min[0], h: frame.max[1] - frame.min[1],
  })), [graph])

  const content = useMemo(() => {
    const all = [...boxes.values()]
    if (!all.length) return { x: 0, y: 0, w: 1, h: 1 }
    const x0 = Math.min(...all.map((box) => box.x - box.w / 2), ...frames.map((frame) => frame.x)) - 40
    const x1 = Math.max(...all.map((box) => box.x + box.w / 2), ...frames.map((frame) => frame.x + frame.w)) + 40
    const y0 = Math.min(...all.map((box) => box.y - box.h / 2), ...frames.map((frame) => frame.y)) - 60
    const y1 = Math.max(...all.map((box) => box.y + box.h / 2), ...frames.map((frame) => frame.y + frame.h)) + 50
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 }
  }, [boxes, frames])

  const fit = useCallback(() => {
    const element = viewport.current
    if (!element) return
    const { width, height } = element.getBoundingClientRect()
    // Keep clear of the floating header above and the footer strip below.
    const top = 150
    const bottom = 70
    const k = Math.min(1.25, Math.max(0.12, Math.min((width - 120) / content.w, (height - top - bottom) / content.h)))
    setView({ k, x: (width - content.w * k) / 2 - content.x * k, y: top + Math.max(0, (height - top - bottom - content.h * k) / 2) - content.y * k })
  }, [content])
  useEffect(() => { fit() }, [fitRequest, modelId, fit])

  useEffect(() => {
    const element = viewport.current
    if (!element) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const rect = element.getBoundingClientRect()
      const px = event.clientX - rect.left
      const py = event.clientY - rect.top
      setView((current) => {
        const k = Math.min(3, Math.max(0.08, current.k * Math.exp(-event.deltaY * 0.0015)))
        return { k, x: px - ((px - current.x) / current.k) * k, y: py - ((py - current.y) / current.k) * k }
      })
    }
    element.addEventListener('wheel', onWheel, { passive: false })
    return () => element.removeEventListener('wheel', onWheel)
  }, [])

  return <div ref={viewport} className="diagram-view"
    onPointerDown={(event) => { drag.current = { x: event.clientX, y: event.clientY, vx: view.x, vy: view.y, moved: false } }}
    onPointerMove={(event) => {
      const start = drag.current
      if (!start || event.buttons === 0) return
      const dx = event.clientX - start.x
      const dy = event.clientY - start.y
      if (Math.abs(dx) + Math.abs(dy) > 3) start.moved = true
      if (start.moved) setView((current) => ({ ...current, x: start.vx + dx, y: start.vy + dy }))
    }}
    onPointerUp={() => { setTimeout(() => { drag.current = null }, 0) }}>
    <svg width="100%" height="100%">
      <defs>
        {Object.entries(edgeColor).map(([type, color]) => <marker key={type} id={`arrow-${type}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill={color} /></marker>)}
        <pattern id="diagram-dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" style={{ fill: 'var(--dot)' }} /></pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#diagram-dots)" />
      <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
        {frames.map(({ frame, segment, x, y, w, h }) => {
          const color = typeColor[frame.node.type] ?? '#94a3b8'
          const active = selected === frame.id || hovered === frame.id
          return <g key={`${frame.id}:${segment}`} className="diagram-frame">
            <rect x={x} y={y} width={w} height={h} rx={14} fill={color} fillOpacity={active ? 0.09 : 0.035} stroke={color} strokeOpacity={active ? 0.9 : 0.35} strokeDasharray={frame.depth > 1 ? '5 4' : undefined} />
            {segment === 0 && <text x={x + 12} y={y + 17} className="diagram-frame-label" fill={color}
              onClick={(event) => { event.stopPropagation(); if (!drag.current?.moved) selectNode(frame.id) }}
              onDoubleClick={(event) => { event.stopPropagation(); toggleCollapsed(frame.id, true) }}>
              ⊟ {localizedNode(frame.node, locale, modelId).name}{frame.node.paramCount ? `  ·  ${formatCount(frame.node.paramCount)}` : ''}
            </text>}
          </g>
        })}
        {graph.edges.map((edge) => {
          const source = graph.byId.get(edge.source)
          const target = graph.byId.get(edge.target)
          if (!source || !target) return null
          const type = edge.type ?? 'data'
          const incident = hops ? hops.has(`${edge.source}>${edge.target}`) : Boolean(focus?.has(edge.source) && focus?.has(edge.target))
          const muted = trace && focus && !incident
          if (edge.stub) {
            // A far use of a broadcast value: a short labelled tap into the card from below.
            const b = boxes.get(target.id)!
            const x = stubX(b, edge)
            const bottom = b.y + b.h / 2
            const name = typeof edge.stub.label === 'string' ? edge.stub.label : localizedNode(edge.stub.label, locale, modelId).name
            return <g key={edge.id} className="diagram-stub" opacity={muted ? 0.15 : 1}>
              <path d={diagramEdgePath(graph, boxes, edge)} fill="none" stroke={edgeColor[type]} strokeWidth={incident ? 2.2 : 1.4} strokeOpacity={incident ? 1 : 0.75} strokeDasharray="4 3" markerEnd={`url(#arrow-${type})`} />
              <text x={x} y={bottom + STUB + 11} textAnchor="middle" fill={edgeColor[type]} fontWeight={incident ? 700 : 600}>{name.length > 16 ? `${name.slice(0, 15)}…` : name}</text>
            </g>
          }
          const d = diagramEdgePath(graph, boxes, edge)
          const line = <path key={edge.id} d={d} fill="none" stroke={edgeColor[type]} strokeWidth={incident ? 2.4 : 1.5}
            strokeOpacity={muted ? 0.1 : incident ? 1 : 0.6} strokeDasharray={type === 'condition' || type === 'training' || type === 'feedback' ? '6 5' : undefined} markerEnd={`url(#arrow-${type})`} />
          if (!edge.transform || muted) return line
          // A reshape on the way: a small diamond at the middle of the line, named when the line is in focus.
          const [mx, my] = midpoint(samplePath(d))
          const label = `${edge.transform.ops.map((op) => op === 'index' ? '[…]' : op).join(' · ')}  ${edge.transform.shape.join('×')}`
          return <g key={edge.id}>{line}<g className="diagram-reshape" transform={`translate(${mx} ${my})`}>
            <rect x={-4.5} y={-4.5} width={9} height={9} transform="rotate(45)" fill={edgeColor[type]} />
            <title>{label}</title>
            {incident && <text y={-10} textAnchor="middle" fill={edgeColor[type]}>{label}</text>}
          </g></g>
        })}
        {fanouts(graph).map(({ source, count, type }) => {
          const a = boxes.get(source)!
          const incident = Boolean(focus?.has(source))
          const x = a.x + a.w / 2
          return <g key={`${source}>taps`} className="diagram-stub" opacity={trace && focus && !incident ? 0.15 : 1}>
            <path d={`M ${x} ${a.y} L ${x + STUB} ${a.y}`} fill="none" stroke={edgeColor[type]} strokeWidth={incident ? 2.2 : 1.4} strokeDasharray="4 3" />
            <text x={x + STUB + 4} y={a.y + 3.5} fill={edgeColor[type]} fontWeight={600}>{locale === 'zh' ? `→ ${count} 处` : `→ ${count} uses`}</text>
          </g>
        })}
        {graph.nodes.map((item) => {
          const box = boxes.get(item.id)!
          const { node } = item
          const color = typeColor[node.type] ?? '#94a3b8'
          const active = selected === item.id || hovered === item.id
          const muted = Boolean(trace && focus && !focus.has(item.id))
          const name = localizedNode(node, locale, modelId).name
          const shape = node.outputShape?.join('×')
          const showShape = labels === 'all' || labels === 'smart' && box.kind !== 'op' && box.kind !== 'param'
          return <g key={item.id} transform={`translate(${box.x} ${box.y})`} opacity={muted ? 0.18 : 1} className="diagram-node"
            onClick={(event) => {
              event.stopPropagation()
              if (drag.current?.moved) return
              if (event.shiftKey && selected) useStudyStore.getState().setPathTarget(item.id)
              else selectNode(item.id)
            }}
            onDoubleClick={(event) => { event.stopPropagation(); if (item.collapsed) toggleCollapsed(item.id, false) }}
            onPointerEnter={() => hoverNode(item.id)} onPointerLeave={() => hoverNode(null)}>
            {box.kind === 'op' && <>
              <circle r={17} style={{ fill: 'var(--diagram-op)' }} stroke={active ? scene.outline : color} strokeWidth={active ? 2 : 1.4} />
              <text className="diagram-symbol" y={5}>{opSymbol[node.className ?? node.name] ?? opSymbol[node.name] ?? '•'}</text>
              {labels !== 'none' && <text className="diagram-caption" y={32}>{name}</text>}
            </>}
            {box.kind === 'param' && <>
              <rect x={-11} y={-11} width={22} height={22} transform="rotate(45)" fill={color} fillOpacity={0.25} stroke={color} />
              {labels !== 'none' && <text className="diagram-caption" y={30}>{name}</text>}
            </>}
            {(box.kind === 'card' || box.kind === 'io' || box.kind === 'group') && <>
              {box.kind === 'group' && <rect x={-box.w / 2 + 5} y={-box.h / 2 + 5} width={box.w} height={box.h} rx={10} style={{ fill: 'var(--diagram-card)' }} stroke={color} strokeOpacity={0.5} />}
              <rect x={-box.w / 2} y={-box.h / 2} width={box.w} height={box.h} rx={box.kind === 'io' ? 22 : 10}
                style={{ fill: box.kind === 'io' ? 'var(--diagram-io)' : 'var(--diagram-card)', filter: box.kind === 'card' || box.kind === 'group' ? 'var(--diagram-shadow)' : undefined }} stroke={active ? scene.outline : color} strokeOpacity={active ? 1 : 0.75} strokeWidth={active ? 2 : 1.2}
                strokeDasharray={box.kind === 'io' ? '4 3' : undefined} />
              {box.kind !== 'io' && <rect x={-box.w / 2} y={-box.h / 2 + 8} width={3} height={box.h - 16} rx={1.5} fill={color} />}
              <text className="diagram-type" x={-box.w / 2 + 12} y={-box.h / 2 + 15} fill={color}>{node.className && !item.card ? node.className : typeName(locale, node.type)}</text>
              <text className="diagram-name" x={-box.w / 2 + 12} y={showShape && shape ? 3 : 7}>{name.length > 18 ? `${name.slice(0, 17)}…` : name}</text>
              {showShape && shape && <text className="diagram-shape" x={-box.w / 2 + 12} y={box.h / 2 - 9}>{shape.length > 22 ? `${shape.slice(0, 21)}…` : shape}</text>}
              {box.kind === 'group' && <text className="diagram-badge" x={box.w / 2 - 14} y={-box.h / 2 + 15}>＋</text>}
            </>}
            {marks?.get(item.id) && <g className={`diagram-diff ${marks.get(item.id)}`} transform={`translate(${-box.w / 2} ${-box.h / 2})`}>
              <circle r={8} /><text y={3.5}>{marks.get(item.id) === 'added' ? '+' : 'Δ'}</text>
            </g>}
            {notes?.[item.id] && <g className="diagram-note" transform={`translate(${box.w / 2} ${-box.h / 2})`}><circle r={8} /><text y={3.5}>✎</text><title>{notes[item.id]}</title></g>}
          </g>
        })}
      </g>
    </svg>
  </div>
}

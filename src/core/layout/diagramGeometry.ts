// Geometry of the flat diagram: card sizes, positions and edge paths. Pure functions, so the integrity
// check (scripts/check-graphs.mjs) measures exactly what the diagram draws.
import type { LayoutMetrics, ViewEdge, ViewGraph, ViewNode } from './viewGraph'
import type { ModelNode } from '../model-ir/types'

const opTypes = new Set(['add', 'multiply', 'matmul', 'concat', 'split', 'generic', 'loss', 'reshape', 'optimizer'])

export type BoxKind = 'card' | 'op' | 'param' | 'io' | 'group'
export interface Box { x: number; y: number; w: number; h: number; kind: BoxKind }

export function boxShape(node: ModelNode, collapsed: boolean, card?: boolean): { w: number; h: number; kind: BoxKind } {
  if (collapsed) return { w: 136, h: 54, kind: 'group' }
  if (card) return node.type === 'tensor' || node.type === 'latent' ? { w: 140, h: 52, kind: 'io' } : { w: 152, h: 56, kind: 'card' }
  if (node.type === 'parameter') return { w: 30, h: 30, kind: 'param' }
  if (node.type === 'tensor') return { w: 124, h: 50, kind: 'io' }
  if (opTypes.has(node.type) && !node.paramCount) return { w: 34, h: 34, kind: 'op' }
  return { w: 122, h: 46, kind: 'card' }
}

export const boxOf = (item: ViewNode, at: { x: number; y: number }): Box => ({ ...at, ...boxShape(item.node, item.collapsed, item.card) })

/** A polyline through the points with rounded corners. */
export function roundedPath(points: [number, number][], radius = 16): string {
  let d = `M ${points[0][0]} ${points[0][1]}`
  for (let index = 1; index < points.length - 1; index++) {
    const [px, py] = points[index - 1]
    const [x, y] = points[index]
    const [nx, ny] = points[index + 1]
    const inLength = Math.hypot(x - px, y - py)
    const outLength = Math.hypot(nx - x, ny - y)
    const r = Math.min(radius, inLength / 2, outLength / 2)
    if (r < 0.5) { d += ` L ${x} ${y}`; continue }
    d += ` L ${x - ((x - px) / inLength) * r} ${y - ((y - py) / inLength) * r} Q ${x} ${y} ${x + ((nx - x) / outLength) * r} ${y + ((ny - y) / outLength) * r}`
  }
  const [lx, ly] = points[points.length - 1]
  return `${d} L ${lx} ${ly}`
}

export const LANE = 82
/** Length of a labelled tap (a far use of a broadcast value). */
export const STUB = 26


export const CELL = { w: 196, h: 108 }

export function diagramMetrics(aspect: number): LayoutMetrics {
  // The layout tests each line as the diagram draws it (y flipped back to point up).
  const sample = (edge: ViewEdge, a: ViewNode, b: ViewNode): [number, number][] => {
    const boxes = new Map([a, b].map((item) => [item.id, boxOf(item, { x: item.position[0], y: -item.position[1] })]))
    return samplePath(diagramEdgePath({ kind: 'flow' } as ViewGraph, boxes, edge)).map(([x, y]) => [x, -y])
  }
  return { size: (node, _glyph, collapsed) => { const { w, h } = boxShape(node, collapsed); return [w, h, 0] }, laneHeight: LANE, gap: 34, unit: 40, aspect, rowWidth: [560, 4200], sample, tolerance: 2 }
}

/** Card boxes in diagram coordinates (y grows downward). */
export function diagramBoxes(graph: ViewGraph): Map<string, Box> {
  const manual = graph.kind === 'manual'
  return new Map(graph.nodes.map((item) => [item.id, boxOf(item, manual
    ? { x: item.grid[0] * CELL.w, y: item.grid[1] * CELL.h }
    : { x: item.position[0], y: -item.position[1] })]))
}

/** Where a labelled tap (a far use of a broadcast value) meets its target card. */
export function stubX(box: Box, edge: ViewEdge): number {
  return box.x + ((edge.stub?.index ?? 0) - ((edge.stub?.count ?? 1) - 1) / 2) * Math.min(64, box.w * 0.45)
}

/** SVG path for one edge. */
export function diagramEdgePath(graph: ViewGraph, boxes: Map<string, Box>, edge: ViewEdge): string {
  const a = boxes.get(edge.source)!
  const b = boxes.get(edge.target)!
  const type = edge.type ?? 'data'
  if (edge.stub) {
    const x = stubX(b, edge)
    const bottom = b.y + b.h / 2
    return `M ${x} ${bottom + STUB} L ${x} ${bottom + 2}`
  }
  const route = edge.via
  if (route?.length) {
    // Routed edge: leave and enter each box on the side facing the route.
    const manual = graph.kind === 'manual'
    const via = route.map(([x, y]) => (manual ? [x / 3.35 * CELL.w, -y / 3.1 * CELL.h] : [x, -y]) as [number, number])
    const port = (box: Box, [tx, ty]: [number, number]): [number, number] => Math.abs(tx - box.x) > Math.abs(ty - box.y)
      ? [box.x + Math.sign(tx - box.x) * box.w / 2, box.y]
      : [box.x, box.y + Math.sign(ty - box.y) * box.h / 2]
    return roundedPath([port(a, via[0]), ...via, port(b, via[via.length - 1])])
  }
  if (type === 'residual' || type === 'branch') {
    const sign = type === 'residual' ? -1 : 1
    const sy = a.y + sign * a.h / 2
    const ey = b.y + sign * b.h / 2
    const lift = sign * (28 + Math.min(70, Math.abs(b.x - a.x) * 0.06))
    const peak = (sign < 0 ? Math.min(sy, ey) : Math.max(sy, ey)) + lift
    return `M ${a.x} ${sy} C ${a.x + 20 * Math.sign(b.x - a.x)} ${peak} ${b.x - 20 * Math.sign(b.x - a.x)} ${peak} ${b.x} ${ey}`
  }
  const direction = Math.sign(b.x - a.x)
  if (Math.abs(b.x - a.x) > (a.w + b.w) / 2 + 4) {
    // Rows of the snake run in both directions.
    const sx = a.x + direction * a.w / 2
    const ex = b.x - direction * b.w / 2
    const handle = direction * Math.max(18, Math.abs(ex - sx) * 0.5)
    return `M ${sx} ${a.y} C ${sx + handle} ${a.y} ${ex - handle} ${b.y} ${ex} ${b.y}`
  }
  const down = b.y > a.y
  const sy = a.y + (down ? a.h / 2 : -a.h / 2)
  const ey = b.y + (down ? -b.h / 2 : b.h / 2)
  return `M ${a.x} ${sy} C ${a.x} ${(sy + ey) / 2} ${b.x} ${(sy + ey) / 2} ${b.x} ${ey}`
}

/** Samples an SVG path made of M, L, Q and C commands into a polyline. */
export function samplePath(d: string, steps = 12): [number, number][] {
  const tokens = d.match(/[MLQC]|-?[\d.]+(?:e-?\d+)?/g) ?? []
  const points: [number, number][] = []
  let index = 0
  let current: [number, number] = [0, 0]
  const read = (): [number, number] => [Number(tokens[index++]), Number(tokens[index++])]
  while (index < tokens.length) {
    const command = tokens[index++]
    if (command === 'M' || command === 'L') { current = read(); points.push(current); continue }
    const controls = command === 'Q' ? [read()] : [read(), read()]
    const end = read()
    const p = [current, ...controls, end]
    for (let step = 1; step <= steps; step++) {
      const t = step / steps
      let level = p
      while (level.length > 1) level = level.slice(1).map((point, i) => [level[i][0] + (point[0] - level[i][0]) * t, level[i][1] + (point[1] - level[i][1]) * t] as [number, number])
      points.push(level[0])
    }
    current = end
  }
  return points
}

export type { ViewNode, ModelNode }

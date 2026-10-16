// Integrity of a model graph and of its drawing: broken ports, shape mismatches, overlapping blocks,
// lines through blocks and line crossings. Used by scripts/check-graphs.mjs and by the in-app check.
import type { ModelIR, TensorShape } from '../model-ir/types'
import type { ViewGraph } from './viewGraph'

export interface Rect { x: number; y: number; w: number; h: number }
export type Point = [number, number]

export interface IntegrityReport {
  edges: number
  ports: string[]
  shapes: string[]
  overlaps: string[]
  through: string[]
  crossings: string[]
}

const fmt = (shape?: TensorShape) => shape ? `[${shape.join(', ')}]` : '?'
const same = (a?: TensorShape, b?: TensorShape) => Boolean(a && b && a.length === b.length && a.every((value, index) => String(value) === String(b[index])))

/** Ports and shapes of the model itself, independent of any drawing. */
export function checkStructure(model: ModelIR): Pick<IntegrityReport, 'ports' | 'shapes'> {
  const byId = new Map(model.nodes.map((node) => [node.id, node]))
  const ports: string[] = []
  const shapes: string[] = []
  const seen = new Set<string>()
  if (byId.size !== model.nodes.length) ports.push('duplicate node ids')
  for (const edge of model.edges) {
    const source = byId.get(edge.source)
    const target = byId.get(edge.target)
    if (!source || !target) { ports.push(`${edge.source} → ${edge.target}: missing ${!source ? 'source' : 'target'}`); continue }
    if (edge.source === edge.target) ports.push(`${edge.source}: connects to itself`)
    const key = `${edge.source}>${edge.target}`
    if (seen.has(key)) ports.push(`${key}: duplicate edge`)
    seen.add(key)
    // A traced data edge carries the source's output into the target: the target must have received that shape.
    if (!source.className || !target.className || (edge.type && edge.type !== 'data' && edge.type !== 'residual' && edge.type !== 'branch')) continue
    const received: TensorShape[] = target.inputShape ? [target.inputShape] : []
    const listed = target.parameters?.inputs
    if (typeof listed === 'string') for (const part of listed.split(' + ')) received.push(part.replace(/[[\]]/g, '').split(',').map((value) => value.trim()).filter(Boolean).map((value) => Number.isNaN(Number(value)) ? value : Number(value)))
    // A reshape between the two (view, transpose, rearrange…) is recorded on the edge and drawn on the line.
    const sent = edge.transform?.shape ?? source.outputShape
    if (sent && received.length && !received.some((shape) => same(shape, sent))) {
      shapes.push(`${edge.source} → ${edge.target}: sends ${fmt(sent)}, receives ${received.map(fmt).join(' / ')}`)
    }
  }
  return { ports, shapes }
}

function segmentsCross([a, b]: [Point, Point], [c, d]: [Point, Point]): boolean {
  const orient = (p: Point, q: Point, r: Point) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
  const d1 = orient(c, d, a), d2 = orient(c, d, b), d3 = orient(a, b, c), d4 = orient(a, b, d)
  return d1 * d2 < 0 && d3 * d4 < 0
}

function segmentHitsRect([a, b]: [Point, Point], rect: Rect): boolean {
  const x0 = rect.x - rect.w / 2, x1 = rect.x + rect.w / 2, y0 = rect.y - rect.h / 2, y1 = rect.y + rect.h / 2
  const inside = (p: Point) => p[0] > x0 && p[0] < x1 && p[1] > y0 && p[1] < y1
  if (inside(a) || inside(b)) return true
  const corners: Point[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
  return corners.some((corner, index) => segmentsCross([a, b], [corner, corners[(index + 1) % 4]]))
}

/**
 * Checks one drawing of the graph. `boxes` are the drawn blocks and `lines` the drawn edges as polylines,
 * both in the same 2D coordinates (the front view for the 3D scene).
 */
export function checkDrawing(model: ModelIR, graph: ViewGraph, boxes: Map<string, Rect>, lines: Map<string, Point[]>, tolerance: number): IntegrityReport {
  const { ports, shapes } = checkStructure(model)
  for (const edge of graph.edges) if (!graph.byId.has(edge.source) || !graph.byId.has(edge.target)) ports.push(`view edge ${edge.id}: endpoint not drawn`)

  const overlaps: string[] = []
  const items = [...boxes]
  for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const [ia, a] = items[i]
    const [ib, b] = items[j]
    if (Math.abs(a.x - b.x) < (a.w + b.w) / 2 - tolerance && Math.abs(a.y - b.y) < (a.h + b.h) / 2 - tolerance) overlaps.push(`${ia} ⟷ ${ib}`)
  }

  const segments = new Map<string, [Point, Point][]>()
  for (const [id, points] of lines) segments.set(id, points.slice(1).map((point, index) => [points[index], point]))
  const through: string[] = []
  for (const edge of graph.edges) {
    const parts = segments.get(edge.id)
    if (!parts) continue
    for (const [id, box] of boxes) {
      if (id === edge.source || id === edge.target) continue
      const shrunk = { ...box, w: box.w - tolerance * 2, h: box.h - tolerance * 2 }
      if (shrunk.w > 0 && shrunk.h > 0 && parts.some((part) => segmentHitsRect(part, shrunk))) through.push(`${edge.source} → ${edge.target} crosses ${id}`)
    }
  }

  const crossings: string[] = []
  const edges = graph.edges.filter((edge) => segments.has(edge.id))
  for (let i = 0; i < edges.length; i++) for (let j = i + 1; j < edges.length; j++) {
    const a = edges[i], b = edges[j]
    // Lines that share a block meet there by design.
    if (a.source === b.source || a.target === b.target || a.source === b.target || a.target === b.source) continue
    const sa = segments.get(a.id)!, sb = segments.get(b.id)!
    if (sa.some((x) => sb.some((y) => segmentsCross(x, y)))) crossings.push(`${a.source} → ${a.target} × ${b.source} → ${b.target}`)
  }
  return { edges: graph.edges.length, ports, shapes, overlaps, through, crossings }
}

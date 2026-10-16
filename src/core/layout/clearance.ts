// Last layout pass: every line is tested against the blocks as it will actually be drawn. A line that
// would pass through a block it does not connect to is rerouted around it, choosing among the clear
// detours the one that crosses the fewest other lines.
import type { Vec3, ViewEdge, ViewNode } from './viewGraph'

export type Point2 = [number, number]
/** The drawn path of an edge as a polyline in layout coordinates (x right, y up). */
export type EdgeSampler = (edge: ViewEdge, a: ViewNode, b: ViewNode) => Point2[]
interface Box { x: number; y: number; w: number; h: number }

const orient = (p: Point2, q: Point2, r: Point2) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
export function segmentsCross(a: Point2, b: Point2, c: Point2, d: Point2): boolean {
  return orient(c, d, a) * orient(c, d, b) < 0 && orient(a, b, c) * orient(a, b, d) < 0
}

function hits(a: Point2, b: Point2, box: Box): boolean {
  const x0 = box.x - box.w / 2, x1 = box.x + box.w / 2, y0 = box.y - box.h / 2, y1 = box.y + box.h / 2
  if (box.w <= 0 || box.h <= 0) return false
  const inside = (p: Point2) => p[0] > x0 && p[0] < x1 && p[1] > y0 && p[1] < y1
  if (inside(a) || inside(b)) return true
  const corners: Point2[] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
  return corners.some((corner, index) => segmentsCross(a, b, corner, corners[(index + 1) % 4]))
}

const length = (points: Point2[]) => points.slice(1).reduce((sum, point, index) => sum + Math.hypot(point[0] - points[index][0], point[1] - points[index][1]), 0)

/** One way the graph is drawn: its edge paths, block footprints and how close a line may graze. */
export interface DrawnView {
  sample: EdgeSampler
  footprint?: (item: ViewNode) => [number, number]
  tolerance: number
}

/**
 * Reroutes edges that pass through blocks in any of the views (the 3D scene and the flat diagram of a
 * study map share one graph). `margin` is the clearance a detour keeps.
 */
export function clearEdges(edges: ViewEdge[], byId: Map<string, ViewNode>, views: DrawnView[], margin: number, z = -0.4): ViewEdge[] {
  const layers = views.map((view) => ({
    ...view,
    boxes: new Map<string, Box>([...byId.values()].map((item) => {
      const [w, h] = view.footprint?.(item) ?? [item.size[0], item.size[1]]
      return [item.id, { x: item.position[0], y: item.position[1], w, h }]
    })),
  }))
  const grow = (box: Box, by: number): Box => ({ ...box, w: box.w + by * 2, h: box.h + by * 2 })
  const blockedIn = (layer: typeof layers[number], edge: ViewEdge, points: Point2[], detour: boolean) => {
    const by = detour ? margin * 0.5 : -layer.tolerance
    for (const [id, box] of layer.boxes) {
      if (id === edge.source || id === edge.target) continue
      const area = grow(box, by)
      for (let index = 1; index < points.length; index++) if (hits(points[index - 1], points[index], area)) return true
    }
    // A detour must not double back through its own ends either (the samples next to the ports excepted).
    if (detour) for (const id of [edge.source, edge.target]) {
      const area = grow(layer.boxes.get(id)!, -layer.tolerance)
      for (let index = 3; index < points.length - 2; index++) if (hits(points[index - 1], points[index], area)) return true
    }
    return false
  }
  const crossings = (edge: ViewEdge, points: Point2[], drawn: Map<string, { edge: ViewEdge; points: Point2[] }>) => {
    let count = 0
    for (const other of drawn.values()) {
      if (other.edge.id === edge.id) continue
      const shared = other.edge.source === edge.source || other.edge.target === edge.target || other.edge.source === edge.target || other.edge.target === edge.source
      if (shared) continue
      outer: for (let i = 1; i < points.length; i++) for (let j = 1; j < other.points.length; j++) {
        if (segmentsCross(points[i - 1], points[i], other.points[j - 1], other.points[j])) { count++; break outer }
      }
    }
    return count
  }
  const sampleAll = (edge: ViewEdge, a: ViewNode, b: ViewNode) => layers.map((layer) => layer.sample(edge, a, b))

  // Crossings are counted in the first view.
  const drawn = new Map<string, { edge: ViewEdge; points: Point2[] }>()
  for (const edge of edges) {
    const a = byId.get(edge.source), b = byId.get(edge.target)
    if (a && b) drawn.set(edge.id, { edge, points: layers[0].sample(edge, a, b) })
  }
  // Channels already taken by a detour: later detours keep their distance instead of running on top.
  const usedH: { y: number; x0: number; x1: number }[] = []
  const usedV: { x: number; y0: number; y1: number }[] = []
  const freeH = (y: number, x0: number, x1: number, step: number) => {
    for (let k = 0; k < 6 && usedH.some((used) => Math.abs(used.y - y) < Math.abs(step) * 0.9 && used.x0 < x1 && x0 < used.x1); k++) y += step
    return y
  }
  const freeV = (x: number, y0: number, y1: number, step: number) => {
    for (let k = 0; k < 6 && usedV.some((used) => Math.abs(used.x - x) < Math.abs(step) * 0.9 && used.y0 < y1 && y0 < used.y1); k++) x += step
    return x
  }
  // The largest footprint over all views decides where channels can run.
  const outline = (id: string): Box => layers.map((layer) => layer.boxes.get(id)!).reduce((big, box) => box.w * box.h > big.w * big.h ? box : big)

  /** The best clear detour for an edge, scored by length plus a heavy price per crossing. */
  const detour = (edge: ViewEdge, a: ViewNode, b: ViewNode) => {
    const [ax, ay] = a.position, [bx, by] = b.position
    const [x0, x1] = [Math.min(ax, bx), Math.max(ax, bx)]
    const [y0, y1] = [Math.min(ay, by), Math.max(ay, by)]
    // Horizontal channels just above or below each block in the way, vertical ones beside them.
    const ys = new Set<number>()
    const xs = new Set<number>()
    for (const item of byId.values()) {
      const box = outline(item.id)
      const own = item === a || item === b
      if (own || box.x + box.w / 2 > x0 - margin && box.x - box.w / 2 < x1 + margin) { ys.add(box.y + box.h / 2 + margin * 1.6); ys.add(box.y - box.h / 2 - margin * 1.6) }
      if (own || box.y + box.h / 2 > y0 - margin && box.y - box.h / 2 < y1 + margin) { xs.add(box.x + box.w / 2 + margin * 1.6); xs.add(box.x - box.w / 2 - margin * 1.6) }
    }
    const mid = [(ax + bx) / 2, (ay + by) / 2]
    const rows = [...ys].sort((p, q) => Math.abs(p - mid[1]) - Math.abs(q - mid[1])).slice(0, 16).map((y) => freeH(y, x0, x1, margin * (y >= mid[1] ? 1 : -1)))
    const columns = [...xs].sort((p, q) => Math.abs(p - mid[0]) - Math.abs(q - mid[0])).slice(0, 16).map((x) => freeV(x, y0, y1, margin * (x >= mid[0] ? 1 : -1)))
    const candidates: { via: Point2[]; kind: 'h' | 'v' | 'l'; at: number }[] = [
      { via: [[bx, ay]], kind: 'l', at: 0 },
      { via: [[ax, by]], kind: 'l', at: 0 },
      ...rows.map((y) => ({ via: [[ax, y], [bx, y]] as Point2[], kind: 'h' as const, at: y })),
      ...columns.map((x) => ({ via: [[x, ay], [x, by]] as Point2[], kind: 'v' as const, at: x })),
      // Around a corner: sideways out of the source, along a free channel, into the target from above or below (and back).
      ...columns.flatMap((x) => rows.flatMap((y) => [
        { via: [[x, ay], [x, y], [bx, y]] as Point2[], kind: 'h' as const, at: y },
        { via: [[ax, y], [x, y], [x, by]] as Point2[], kind: 'v' as const, at: x },
      ])),
    ]
    let best: { edge: ViewEdge; points: Point2[]; score: number; crossings: number; length: number; kind: 'h' | 'v' | 'l'; at: number; x0: number; x1: number; y0: number; y1: number } | undefined
    const atEnd = (point: Point2) => [a, b].some((item) => { const box = outline(item.id); return Math.abs(point[0] - box.x) < box.w / 2 && Math.abs(point[1] - box.y) < box.h / 2 })
    for (const candidate of candidates) {
      if (candidate.via.some(atEnd)) continue
      const via = candidate.via.filter((point, index, all) => index === 0 || Math.hypot(point[0] - all[index - 1][0], point[1] - all[index - 1][1]) > 1e-6)
      const routed: ViewEdge = { ...edge, via: via.map(([x, y]) => [x, y, z] as Vec3) }
      const samples = sampleAll(routed, a, b)
      if (samples.some((points, index) => blockedIn(layers[index], routed, points, true))) continue
      const count = crossings(routed, samples[0], drawn)
      const score = length(samples[0]) + count * margin * 40
      if (!best || score < best.score) best = { edge: routed, points: samples[0], score, crossings: count, length: length(samples[0]), kind: candidate.kind, at: candidate.at, x0, x1, y0, y1 }
    }
    return best
  }
  const take = (best: NonNullable<ReturnType<typeof detour>>) => {
    drawn.set(best.edge.id, { edge: best.edge, points: best.points })
    if (best.kind === 'h') usedH.push({ y: best.at, x0: best.x0, x1: best.x1 })
    if (best.kind === 'v') usedV.push({ x: best.at, y0: best.y0, y1: best.y1 })
    return best.edge
  }

  // 1. Nothing may pass through a block.
  const cleared = edges.map((edge) => {
    const a = byId.get(edge.source), b = byId.get(edge.target)
    if (!a || !b || edge.stub) return edge
    if (!sampleAll(edge, a, b).some((points, index) => blockedIn(layers[index], edge, points, false))) return edge
    const best = detour(edge, a, b)
    return best ? take(best) : edge
  })
  // 2. Lines that still cross another take a detour when one crosses fewer lines and is not much longer.
  for (let round = 0; round < 2; round++) {
    const scored = cleared.map((edge, index) => ({ index, count: drawn.has(edge.id) && !edge.stub ? crossings(edge, drawn.get(edge.id)!.points, drawn) : 0 }))
    for (const { index } of scored.filter((item) => item.count > 0).sort((p, q) => q.count - p.count)) {
      const edge = cleared[index]
      const a = byId.get(edge.source)!, b = byId.get(edge.target)!
      const now = crossings(edge, drawn.get(edge.id)!.points, drawn)
      if (!now) continue
      const best = detour({ ...edge, via: undefined }, a, b)
      if (best && best.crossings < now && best.length < length(drawn.get(edge.id)!.points) * 2.5 + margin * 20) cleared[index] = take(best)
    }
  }
  return cleared
}

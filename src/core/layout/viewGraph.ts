// Builds the visible graph (collapsed groups, mapped edges) and a deterministic 3D layout.
//
// Two layouts follow the model's own dataflow instead of one fixed chain:
//  • levels: encoder–decoder networks (U-Net, VAE) step down one band per halving of the
//    spatial resolution and climb back up, so skip connections become level bridges across the U.
//  • flow: everything else runs left → right and folds like a snake at stage boundaries
//    (resolution changes, top-level blocks), so a row never needs a long return line.
// The flat diagram runs the same layout with card-sized metrics, so both views share one structure.
import type { ModelEdge, ModelIR, ModelNode, TensorShape } from '../model-ir/types'
import { clearEdges, type EdgeSampler, type Point2 } from './clearance'
import { edgeCurve } from './edgeCurve'
import { CELL, boxOf, diagramEdgePath, samplePath } from './diagramGeometry'

export type Vec3 = [number, number, number]
export type LayoutKind = 'flow' | 'levels' | 'manual'

export interface ViewNode {
  id: string
  node: ModelNode
  position: Vec3
  /** Glyph size: x = thickness along the flow, y = height, z = depth. */
  size: Vec3
  /** Cell for the flat diagram: [column, lane] (lane grows downward, may be fractional). */
  grid: [number, number]
  rank: number
  /** Vertical lane inside the row (0 = main path). */
  lane: number
  /** Layout segment: the folded row (flow) or the resolution level (levels). */
  row: number
  /** Resolution level for the levels layout (0 = full resolution, -1 = condition band). */
  level?: number
  /** Direction the row runs: 1 = left → right, -1 = right → left. */
  dir: 1 | -1
  collapsed: boolean
  depth: number
  /** Hand-authored study maps draw every node as a labelled card. */
  card?: boolean
}

export interface ViewEdge extends ModelEdge {
  via?: Vec3[]
  /** A far-away use of a value that also feeds something nearby (a timestep embedding, a prompt, a
   *  reused input). Drawn as a short labelled tap into the target instead of a line across the layout. */
  stub?: { label: ModelNode | string; index: number; count: number }
}

export interface ViewFrame {
  id: string
  node: ModelNode
  min: [number, number]
  max: [number, number]
  depth: number
  segment: number
  /** Back face of the members (z), so the panel sits just behind them. */
  zMin: number
}

export interface ViewGraph {
  kind: LayoutKind
  nodes: ViewNode[]
  edges: ViewEdge[]
  frames: ViewFrame[]
  byId: Map<string, ViewNode>
  bounds: { min: Vec3; max: Vec3 }
  /** Maps any model node id to the visible node that represents it. */
  representative: Map<string, string>
  maxDepth: number
}

const numeric = (shape?: TensorShape) => (shape ?? []).map((axis) => typeof axis === 'number' ? axis : NaN)
const sizeOf = (n: number, scale = 0.34, base = 0.55, max = 3.3) => Number.isFinite(n) && n > 0 ? Math.min(max, base + Math.log2(Math.max(1, n)) * scale) : 1.2
const thickOf = (n: number) => Number.isFinite(n) && n > 0 ? Math.min(1.25, 0.14 + Math.log2(Math.max(1, n)) * 0.1) : 0.5

const thinTypes = new Set(['activation', 'dropout', 'norm'])
const opTypes = new Set(['add', 'multiply', 'matmul', 'concat', 'split', 'generic', 'loss', 'reshape', 'optimizer'])

/** Glyph dimensions derived from the tensor a node produces. */
export function glyphSize(node: ModelNode, collapsed = false): Vec3 {
  const shape = numeric(node.outputShape ?? node.inputShape)
  const [batchless, rank] = [shape.length > 1 ? shape.slice(1) : shape, shape.length]
  let size: Vec3
  if (rank >= 5) size = [thickOf(batchless[0]) + 0.2, sizeOf(batchless[2]), sizeOf(batchless[3])]
  else if (rank === 4) size = [thickOf(batchless[0]), sizeOf(batchless[1]), sizeOf(batchless[2])]
  else if (rank === 3) size = [0.42, sizeOf(batchless[0], 0.3), sizeOf(batchless[1], 0.24, 0.4, 2.4)]
  else if (rank === 2) size = [0.36, sizeOf(batchless[0], 0.3), 0.36]
  else size = [0.4, 0.6, 0.4]
  if (Number.isNaN(shape[0]) && !shape.some(Number.isFinite)) size = [0.9, 1.15, 0.9]
  if (collapsed) return [Math.max(1.1, size[0] + 0.6), Math.max(1.3, size[1]), Math.max(1.0, size[2])]
  if (opTypes.has(node.type) && !node.paramCount) return [0.56, 0.56, 0.56]
  if (thinTypes.has(node.type)) return [0.1, size[1], size[2]]
  if (node.type === 'parameter') return [0.3, Math.min(size[1], 1.6), Math.min(size[2], 1.6)]
  return size
}

function childrenMap(model: ModelIR): Map<string | undefined, ModelNode[]> {
  const map = new Map<string | undefined, ModelNode[]>()
  model.nodes.forEach((node) => map.set(node.parentId, [...(map.get(node.parentId) ?? []), node]))
  return map
}

export function groupDepths(model: ModelIR): Map<string, number> {
  const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
  const depths = new Map<string, number>()
  model.nodes.forEach((node) => {
    let depth = 0
    for (let current = node.parentId; current; current = parents.get(current)) depth++
    depths.set(node.id, depth)
  })
  return depths
}

export function groupIds(model: ModelIR): Set<string> {
  return new Set(model.nodes.map((node) => node.parentId).filter((id): id is string => Boolean(id)))
}

const containerClasses = new Set(['Sequential', 'ModuleList', 'ModuleDict'])

/**
 * Collapse rule for an expansion level. Besides depth, groups of the same module class are shown at
 * the same granularity: if one DoubleConv is collapsed, the stem DoubleConv one level higher is too.
 * Plain containers are matched by their parent's class and their own name instead.
 */
export function depthCollapse(model: ModelIR, level: number): (id: string) => boolean {
  const groups = groupIds(model)
  const depths = groupDepths(model)
  const byId = new Map(model.nodes.map((node) => [node.id, node]))
  const signature = (node: ModelNode) => {
    const className = node.className ?? node.type
    if (!containerClasses.has(className) && node.className) return className
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    return `${parent?.className ?? 'root'}>${className}:${node.id.split(/[./]/).pop()}`
  }
  const collapsedKinds = new Set<string>()
  groups.forEach((id) => { if ((depths.get(id) ?? 0) >= level) collapsedKinds.add(signature(byId.get(id)!)) })
  return (id: string) => groups.has(id) && ((depths.get(id) ?? 0) >= level || collapsedKinds.has(signature(byId.get(id)!)))
}

/** Deepest expansion level that keeps the number of visible nodes readable. */
export function autoDepth(model: ModelIR, limit = isEncoderDecoder(model) ? 40 : 70): number {
  const groups = groupIds(model)
  const depths = groupDepths(model)
  const maxDepth = Math.max(0, ...[...groups].map((id) => depths.get(id) ?? 0)) + 1
  for (let level = maxDepth; level >= 0; level--) {
    const count = visibleCount(model, depthCollapse(model, level))
    if (count > limit) continue
    // Expanding a level that only wraps single blocks in another frame adds clutter, not detail.
    while (level > 0 && visibleCount(model, depthCollapse(model, level - 1)) === count) level--
    return level
  }
  return 0
}

/** Spatial height of a feature map ([B, C, H, W] or [B, C, T, H, W]); NaN for anything else. */
function spatialHeight(node: ModelNode): number {
  const shape = numeric(node.outputShape)
  const height = shape.length === 4 ? shape[2] : shape.length === 5 ? shape[3] : NaN
  return Number.isFinite(height) && height > 1 ? height : NaN
}

/**
 * True for networks whose feature maps shrink through three or more resolutions and then grow back
 * (U-Net, autoencoders): they get the U-shaped levels layout.
 */
export function isEncoderDecoder(model: ModelIR): boolean {
  if (model.layout !== 'code') return false
  const groups = groupIds(model)
  const leaves = model.nodes.filter((node) => !groups.has(node.id) && node.type !== 'parameter')
  const heights = leaves.map(spatialHeight)
  const spatial = heights.filter(Number.isFinite)
  if (spatial.length < 6 || spatial.length < leaves.length * 0.45) return false
  const top = Math.max(...spatial)
  const levels = spatial.map((height) => Math.round(Math.log2(top / height)))
  const deepest = Math.max(...levels)
  if (new Set(levels).size < 3) return false
  const turn = levels.indexOf(deepest)
  return levels.slice(turn).some((level) => level <= deepest - 2) && levels[levels.length - 1] <= 1
}

function visibleCount(model: ModelIR, isCollapsed: (id: string) => boolean): number {
  const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
  const groups = groupIds(model)
  return model.nodes.filter((node) => {
    for (let current = node.parentId; current; current = parents.get(current)) if (isCollapsed(current)) return false
    return !groups.has(node.id) || isCollapsed(node.id)
  }).length
}

const MAX_ROW_WIDTH = 64
const GAP = 1.05
/** Free lanes between folded rows and between resolution bands: room for routed edges and frame labels. */
const ROW_GAP = 1.9
const BAND_GAP = 0.75

/** Units for one layout pass: the 3D scene works in glyph sizes, the flat diagram in card pixels. */
export interface LayoutMetrics {
  /** Block footprint: [width along the flow, height, depth]. */
  size: (node: ModelNode, glyph: Vec3, collapsed: boolean) => Vec3
  /** Height of one lane; by default the tallest block plus a margin. */
  laneHeight?: number
  /** Gap between columns. */
  gap: number
  /** Scale of route margins (1 in the scene). */
  unit: number
  /** Preferred width : height of a folded layout. */
  aspect: number
  /** Allowed width of one folded row. */
  rowWidth: [number, number]
  /** The drawn path of an edge (x right, y up); by default the 3D curve. Used to keep lines off blocks. */
  sample?: EdgeSampler
  /** How far a line may graze a block (3D units by default). */
  tolerance?: number
}

const sceneSample: EdgeSampler = (edge, a, b) => edgeCurve(edge, a, b).getPoints(32).map((point) => [point.x, point.y] as Point2)

export const sceneMetrics: LayoutMetrics = { size: (_node, glyph) => glyph, gap: GAP, unit: 1, aspect: 2.2, rowWidth: [22, MAX_ROW_WIDTH] }

/** Abstract placement before conversion to coordinates. */
interface Slot { row: number; col: number; g: number; x: number; dir: 1 | -1; level?: number; lane?: number }

interface LayoutInput {
  visible: ModelNode[]
  rank: Map<string, number>
  lane: Map<string, number>
  sizes: Map<string, Vec3>
  byRank: Map<number, ModelNode[]>
  ranks: number[]
  predecessors: Map<string, string[]>
  successors: Map<string, string[]>
  parents: Map<string, string | undefined>
  laneHeight: number
  metrics: LayoutMetrics
}

const columnWidths = ({ ranks, byRank, sizes }: LayoutInput) => new Map(ranks.map((r) => [r, Math.max(...byRank.get(r)!.map((node) => sizes.get(node.id)![0]))]))

/** U layout: one horizontal band per spatial resolution, columns in dataflow order. */
function levelSlots(input: LayoutInput): Map<string, Slot> {
  const { visible, lane, ranks, predecessors, successors } = input
  const heights = new Map(visible.map((node) => [node.id, spatialHeight(node)]))
  const top = Math.max(...[...heights.values()].filter(Number.isFinite))
  const level = new Map<string, number>()
  visible.forEach((node) => {
    const height = heights.get(node.id)!
    if (Number.isFinite(height)) level.set(node.id, Math.max(0, Math.round(Math.log2(top / height))))
  })
  // Nearest feature maps reached by walking the graph (breadth first) from a non-spatial node.
  const nearest = (start: string, next: Map<string, string[]>) => {
    const found = new Set<number>()
    const seen = new Set([start])
    let frontier = next.get(start) ?? []
    while (frontier.length && !found.size) {
      const following: string[] = []
      for (const id of frontier) {
        if (seen.has(id)) continue
        seen.add(id)
        if (Number.isFinite(heights.get(id))) found.add(level.get(id)!)
        else following.push(...(next.get(id) ?? []))
      }
      frontier = following
    }
    return found
  }
  // Vectors sit with the feature maps around them; a bottleneck vector (a VAE latent) takes the deeper
  // side. One that feeds several levels (a timestep embedding) goes to a condition band above the U.
  const constants: ModelNode[] = []
  visible.forEach((node) => {
    if (level.has(node.id)) return
    if (!(predecessors.get(node.id) ?? []).length && node.className !== 'Input') { constants.push(node); return }
    const forward = nearest(node.id, successors)
    if (forward.size > 1) { level.set(node.id, -1); return }
    level.set(node.id, Math.max(0, ...forward, ...nearest(node.id, predecessors)))
  })
  // Constants (parameters, sampled noise) sit beside their consumer.
  constants.reverse().forEach((node) => level.set(node.id, Math.max(-1, ...(successors.get(node.id) ?? []).map((id) => level.get(id) ?? 0))))

  // Columns: a step to another band reuses the column (a vertical arrow, as in the U-Net paper);
  // inside a band every rank moves one column right. Side inputs that arrive from another band
  // (a block's timestep projection, sampled noise) are placed afterwards, right beside their consumer.
  const sameBand = (a: string, b: string) => level.get(a) === level.get(b)
  const isSide = (node: ModelNode) => !Number.isFinite(heights.get(node.id)) && level.get(node.id)! >= 0
    && (predecessors.get(node.id) ?? []).every((id) => !sameBand(id, node.id))
    && (successors.get(node.id) ?? []).some((id) => sameBand(id, node.id))
  const col = new Map<string, number>()
  const lastInBand = new Map<number, number>()
  const sides: ModelNode[] = []
  for (const r of ranks) {
    const nodes = input.byRank.get(r)!.filter((node) => isSide(node) ? (sides.push(node), false) : true)
    const perBand = new Map<number, number>()
    nodes.forEach((node) => {
      const band = level.get(node.id)!
      const fromPreds = (predecessors.get(node.id) ?? []).filter((id) => col.has(id)).map((id) => col.get(id)! + (sameBand(id, node.id) ? 1 : 0))
      const wanted = Math.max(fromPreds.length ? Math.max(...fromPreds) : 0, (lastInBand.get(band) ?? -1) + 1)
      perBand.set(band, Math.max(perBand.get(band) ?? 0, wanted))
    })
    nodes.forEach((node) => col.set(node.id, perBand.get(level.get(node.id)!)!))
    perBand.forEach((value, band) => lastInBand.set(band, value))
  }
  sides.forEach((node) => col.set(node.id, Math.max(0, Math.min(...(successors.get(node.id) ?? []).filter((id) => sameBand(id, node.id) && col.has(id)).map((id) => col.get(id)!)) - 1)))

  // Lanes inside each band: the barycenter of same-band predecessors keeps each band's path straight;
  // side inputs stack above it.
  const bandLane = new Map<string, number>()
  const cells = new Map<string, ModelNode[]>()
  visible.forEach((node) => { const key = `${level.get(node.id)}:${col.get(node.id)}`; cells.set(key, [...(cells.get(key) ?? []), node]) })
  ;[...cells.values()].sort((a, b) => col.get(a[0].id)! - col.get(b[0].id)!).forEach((nodes) => {
    const used = new Set<number>()
    nodes.map((node) => {
      const preds = (predecessors.get(node.id) ?? []).filter((id) => sameBand(id, node.id) && bandLane.has(id))
      const side = !preds.length && nodes.length > 1
      return { node, side, bary: preds.length ? preds.reduce((sum, id) => sum + bandLane.get(id)!, 0) / preds.length : side ? -1 : 0 }
    }).sort((a, b) => a.bary - b.bary).forEach((item) => {
      let target = Math.round(item.bary)
      let offset = 0
      while (used.has(target + offset)) offset = offset > 0 ? -offset : -offset + 1
      target += offset
      used.add(target)
      bandLane.set(item.node.id, target)
    })
  })
  void lane

  const bands = [...new Set(level.values())].sort((a, b) => a - b)
  const base = new Map<number, number>()
  let cursor = 0
  for (const band of bands) {
    const lanes = visible.filter((node) => level.get(node.id) === band).map((node) => bandLane.get(node.id)!)
    const low = Math.min(...lanes)
    base.set(band, cursor - low)
    cursor += Math.max(...lanes) - low + 1 + BAND_GAP
  }

  const columns = Math.max(0, ...col.values()) + 1
  const widths = Array.from({ length: columns }, () => 0.3 * input.metrics.unit)
  visible.forEach((node) => { widths[col.get(node.id)!] = Math.max(widths[col.get(node.id)!], input.sizes.get(node.id)![0]) })
  const columnX = [0]
  for (let index = 1; index < columns; index++) columnX.push(columnX[index - 1] + widths[index - 1] / 2 + input.metrics.gap * 1.6 + widths[index] / 2)
  return new Map(visible.map((node) => {
    const band = level.get(node.id)!
    return [node.id, { row: band, level: band, lane: bandLane.get(node.id)!, col: col.get(node.id)!, x: columnX[col.get(node.id)!], g: base.get(band)! + bandLane.get(node.id)!, dir: 1 }]
  }))
}

/** Snake layout: rows break at stage boundaries and alternate direction, so each row ends where the next begins. */
function flowSlots(input: LayoutInput): Map<string, Slot> {
  const { visible, rank, lane, byRank, ranks, predecessors, parents, laneHeight, metrics } = input
  const GAP = metrics.gap
  const widths = columnWidths(input)
  const totalWidth = ranks.reduce((sum, r) => sum + widths.get(r)! + GAP, 0) - GAP
  const laneValues = [...lane.values()]
  const rowHeight = (Math.max(0, ...laneValues) - Math.min(0, ...laneValues) + 1) * laneHeight
  // Pick the number of rows whose overall shape is closest to a wide screen (about 2.2 : 1).
  let rowCount = 1
  let bestShape = Infinity
  for (let count = 1; count <= 8; count++) {
    const shape = Math.abs(Math.log(totalWidth / count / (count * rowHeight + (count - 1) * ROW_GAP * laneHeight) / metrics.aspect))
    if (shape < bestShape) { bestShape = shape; rowCount = count }
  }
  const rowWidth = rowCount === 1 ? Infinity : Math.min(metrics.rowWidth[1], Math.max(metrics.rowWidth[0], (totalWidth / rowCount) * 1.04))

  // How good a place each column boundary is for a fold: resolution changes and top-level module
  // boundaries are preferred; never separate a constant (a parameter, a created tensor) from its consumer.
  const main = (r: number) => [...byRank.get(r)!].sort((a, b) => Math.abs(lane.get(a.id)!) - Math.abs(lane.get(b.id)!))[0]
  const ancestry = (id: string) => { const chain: string[] = []; for (let current = parents.get(id); current; current = parents.get(current)) chain.unshift(current); return chain }
  const priority = ranks.map((r, index) => {
    if (index === 0) return -Infinity
    if (byRank.get(r)!.every((node) => !(predecessors.get(node.id) ?? []).length && node.className !== 'Input')) return -Infinity
    const a = main(ranks[index - 1])
    const b = main(r)
    let score = 0
    const ha = spatialHeight(a)
    const hb = spatialHeight(b)
    if (Number.isFinite(ha) && Number.isFinite(hb) && ha !== hb) score = 3
    const [ca, cb] = [ancestry(a.id), ancestry(b.id)]
    if (ca[0] !== cb[0]) score = Math.max(score, 2)
    else if (ca[1] !== cb[1]) score = Math.max(score, 1)
    return score
  })

  const rows: [number, number][] = []
  let start = 0
  while (start < ranks.length) {
    let width = widths.get(ranks[start])!
    let end = start + 1
    while (end < ranks.length && width + GAP + widths.get(ranks[end])! <= rowWidth) { width += GAP + widths.get(ranks[end])!; end++ }
    if (end >= ranks.length) { rows.push([start, ranks.length]); break }
    let best = -1
    let bestScore = -Infinity
    let offset = widths.get(ranks[start])! + GAP
    for (let index = start + 1; index <= end; index++) {
      const fraction = offset / rowWidth
      if (fraction >= 0.55 && priority[index] > -Infinity) {
        const score = priority[index] + fraction * 0.8
        if (score >= bestScore) { bestScore = score; best = index }
      }
      if (index < ranks.length) offset += widths.get(ranks[index])! + GAP
    }
    if (best < 0) best = end
    rows.push([start, best])
    start = best
  }
  // A short tail row would leave one block stranded: fold it into the row before.
  if (rows.length > 1) {
    const [tailStart, tailEnd] = rows[rows.length - 1]
    const tail = ranks.slice(tailStart, tailEnd).reduce((sum, r) => sum + widths.get(r)! + GAP, 0)
    if (tail < Math.min(rowWidth, totalWidth) * 0.3) { rows.pop(); rows[rows.length - 1][1] = tailEnd }
  }

  const forwardX = new Map<number, number>()
  const forwardCol = new Map<number, number>()
  const rowOf = new Map<number, number>()
  let maxWidth = 0
  let maxColumns = 0
  rows.forEach(([from, to], row) => {
    let x = 0
    for (let index = from; index < to; index++) {
      const r = ranks[index]
      if (index > from) x += widths.get(ranks[index - 1])! / 2 + GAP + widths.get(r)! / 2
      forwardX.set(r, x)
      forwardCol.set(r, index - from)
      rowOf.set(r, row)
    }
    maxWidth = Math.max(maxWidth, x)
    maxColumns = Math.max(maxColumns, to - from)
  })
  const rowLanes = rows.map(([from, to]) => {
    const lanes = ranks.slice(from, to).flatMap((r) => byRank.get(r)!.map((node) => lane.get(node.id)!))
    return [Math.min(...lanes), Math.max(...lanes)] as const
  })
  const base: number[] = []
  rowLanes.forEach(([low], row) => base.push(row === 0 ? -low : base[row - 1] + rowLanes[row - 1][1] + 1 + ROW_GAP - low))
  return new Map(visible.map((node) => {
    const r = rank.get(node.id)!
    const row = rowOf.get(r)!
    const dir: 1 | -1 = row % 2 === 0 ? 1 : -1
    return [node.id, {
      row, dir,
      col: dir === 1 ? forwardCol.get(r)! : maxColumns - 1 - forwardCol.get(r)!,
      x: dir === 1 ? forwardX.get(r)! : maxWidth - forwardX.get(r)!,
      g: base[row] + lane.get(node.id)!,
    }]
  }))
}

export function buildViewGraph(model: ModelIR, isCollapsed: (id: string) => boolean, metrics: LayoutMetrics = sceneMetrics): ViewGraph {
  if (model.layout !== 'code') return manualGraph(model)
  const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
  const groups = groupIds(model)
  const depths = groupDepths(model)
  const children = childrenMap(model)
  const representative = new Map<string, string>()
  const visible: ModelNode[] = []
  model.nodes.forEach((node) => {
    let top: string | undefined
    for (let current = node.parentId; current; current = parents.get(current)) if (isCollapsed(current)) top = current
    if (top) { representative.set(node.id, top); return }
    representative.set(node.id, node.id)
    if (!groups.has(node.id) || isCollapsed(node.id)) visible.push(node)
  })
  const order = new Map(model.nodes.map((node, index) => [node.id, index]))

  const edgeMap = new Map<string, ViewEdge>()
  for (const edge of model.edges) {
    const source = representative.get(edge.source)
    const target = representative.get(edge.target)
    if (!source || !target || source === target) continue
    const key = `${source}>${target}`
    const existing = edgeMap.get(key)
    if (!existing) edgeMap.set(key, { ...edge, id: key, source, target, transform: source === edge.source && target === edge.target ? edge.transform : undefined })
    else if ((existing.type ?? 'data') !== 'data' && (edge.type ?? 'data') === 'data') existing.type = 'data'
  }
  const edges = [...edgeMap.values()]
  const predecessors = new Map<string, string[]>()
  const successors = new Map<string, string[]>()
  edges.forEach((edge) => {
    predecessors.set(edge.target, [...(predecessors.get(edge.target) ?? []), edge.source])
    successors.set(edge.source, [...(successors.get(edge.source) ?? []), edge.target])
  })

  // Longest-path ranks in trace order (the trace is already topologically sorted).
  visible.sort((a, b) => (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0))
  const rank = new Map<string, number>()
  for (const node of visible) {
    const preds = predecessors.get(node.id) ?? []
    rank.set(node.id, preds.length ? Math.max(...preds.map((id) => (rank.get(id) ?? 0) + 1)) : 0)
  }
  // Side branches computed only from inputs and constants (parameters, a timestep embedding, a loss target
  // such as noise − actions) sit right before their first consumer instead of at the far left, so they
  // never need a long line. Branches off the main path (a ResNet shortcut) keep their place at the fork.
  const floating = new Map<string, number>()
  for (const node of visible) {
    const preds = predecessors.get(node.id) ?? []
    const height = Math.max(0, ...preds.map((id) => (floating.get(id) ?? Infinity) + 1))
    if (height <= 4) floating.set(node.id, height)
  }
  for (const node of [...visible].reverse()) {
    const succ = successors.get(node.id) ?? []
    if (floating.has(node.id) && succ.length) rank.set(node.id, Math.max(rank.get(node.id)!, Math.min(...succ.map((id) => rank.get(id) ?? 0)) - 1))
  }

  const sizes = new Map(visible.map((node) => [node.id, metrics.size(node, glyphSize(node, groups.has(node.id)), groups.has(node.id))]))
  const byRank = new Map<number, ModelNode[]>()
  visible.forEach((node) => byRank.set(rank.get(node.id)!, [...(byRank.get(rank.get(node.id)!) ?? []), node]))
  const ranks = [...byRank.keys()].sort((a, b) => a - b)

  // Lanes: barycenter of predecessors keeps the main path straight; side inputs stack above.
  const lane = new Map<string, number>()
  for (const r of ranks) {
    const nodes = byRank.get(r)!
    const scored = nodes.map((node) => {
      const preds = (predecessors.get(node.id) ?? []).filter((id) => lane.has(id))
      const isSource = !(predecessors.get(node.id) ?? []).length && node.className !== 'Input'
      const bary = preds.length ? preds.reduce((sum, id) => sum + lane.get(id)!, 0) / preds.length : isSource ? -1 : 0
      return { node, bary, isSource }
    }).sort((a, b) => a.bary - b.bary || (order.get(a.node.id) ?? 0) - (order.get(b.node.id) ?? 0))
    const used = new Set<number>()
    for (const item of scored) {
      let target = Math.round(item.bary)
      if (item.isSource) target = Math.min(target, -1)
      let offset = 0
      while (used.has(target + offset)) offset = offset > 0 ? -offset : -offset + 1
      used.add(target + offset)
      lane.set(item.node.id, target + offset)
    }
  }

  if (!isEncoderDecoder(model)) untangleLanes(visible, rank, lane, edges)

  const maxHeight = Math.max(1.4, ...[...sizes.values()].map((size) => size[1]))
  const laneHeight = metrics.laneHeight ?? Math.min(3.6, maxHeight + 0.9)
  const kind: LayoutKind = isEncoderDecoder(model) ? 'levels' : 'flow'
  const input: LayoutInput = { visible, rank, lane, sizes, byRank, ranks, predecessors, successors, parents, laneHeight, metrics }
  const slots = kind === 'levels' ? levelSlots(input) : flowSlots(input)

  const viewNodes: ViewNode[] = visible.map((node) => {
    const slot = slots.get(node.id)!
    return {
      id: node.id, node, rank: rank.get(node.id)!, lane: slot.lane ?? lane.get(node.id)!, row: slot.row, level: slot.level, dir: slot.dir,
      collapsed: groups.has(node.id), depth: depths.get(node.id) ?? 0, size: sizes.get(node.id)!,
      position: [slot.x, -slot.g * laneHeight, 0], grid: [slot.col, slot.g],
    }
  })
  const byId = new Map(viewNodes.map((item) => [item.id, item]))
  const descendantsOf = (id: string): ViewNode[] => (children.get(id) ?? []).flatMap((child) => byId.has(child.id) ? [byId.get(child.id)!] : descendantsOf(child.id))
  // A tap is named after the module whose output it is (text_encoder rather than its final norm).
  const nodeById = new Map(model.nodes.map((node) => [node.id, node]))
  const owner = (id: string): ModelNode | string => {
    let current = nodeById.get(id)!
    // An operation result is named by its Python variable (c = t_emb + y_emb → "c").
    if (current.variable && !current.paramCount && current.className !== 'Input' && current.type !== 'tensor') return current.variable
    for (let parent = current.parentId; parent; parent = nodeById.get(parent)?.parentId) {
      const members = descendantsOf(parent)
      const last = members.reduce<ViewNode | undefined>((best, item) => !best || (order.get(item.id) ?? 0) > (order.get(best.id) ?? 0) ? item : best, undefined)
      if (last?.id !== id) break
      current = nodeById.get(parent)!
    }
    return current
  }
  const routed = clearEdges(routeEdges(kind, markStubs(kind, edges, byId, owner), viewNodes, byId, laneHeight, metrics.unit), byId,
    [{ sample: metrics.sample ?? sceneSample, tolerance: metrics.tolerance ?? 0.04 }], 0.28 * metrics.unit)

  // Frames outline expanded groups, split per row (flow) or per resolution band (levels).
  const frames: ViewFrame[] = []
  const maxGroupDepth = Math.max(0, ...[...groups].map((id) => depths.get(id) ?? 0))
  for (const id of groups) {
    if (representative.get(id) !== id || isCollapsed(id)) continue
    const members = descendantsOf(id)
    if (!members.length) continue
    const node = model.nodes.find((item) => item.id === id)!
    const depth = depths.get(id) ?? 0
    const pad = (0.32 + (maxGroupDepth - depth) * 0.26) * metrics.unit
    const byRow = new Map<number, ViewNode[]>()
    members.forEach((member) => byRow.set(member.row, [...(byRow.get(member.row) ?? []), member]))
    let segment = 0
    for (const items of byRow.values()) {
      frames.push({
        id, node, depth, segment: segment++,
        min: [Math.min(...items.map((item) => item.position[0] - item.size[0] / 2)) - pad, Math.min(...items.map((item) => item.position[1] - item.size[1] / 2)) - pad],
        max: [Math.max(...items.map((item) => item.position[0] + item.size[0] / 2)) + pad, Math.max(...items.map((item) => item.position[1] + item.size[1] / 2)) + pad + (0.95 + (maxGroupDepth - depth) * 1.05) * metrics.unit],
        zMin: Math.min(...items.map((item) => item.position[2] - item.size[2] / 2)),
      })
    }
  }

  // Sibling frames side by side (two stages sharing a row) meet halfway instead of overlapping.
  for (let i = 0; i < frames.length; i++) for (let j = i + 1; j < frames.length; j++) {
    const a = frames[i], b = frames[j]
    if (a.depth !== b.depth) continue
    const overlapX = Math.min(a.max[0], b.max[0]) - Math.max(a.min[0], b.min[0])
    const overlapY = Math.min(a.max[1], b.max[1]) - Math.max(a.min[1], b.min[1])
    if (overlapX <= 0 || overlapY <= 0) continue
    const space = 0.08 * metrics.unit
    if (overlapX < overlapY) {
      const [left, right] = a.min[0] < b.min[0] ? [a, b] : [b, a]
      const middle = (left.max[0] + right.min[0]) / 2
      left.max[0] = middle - space
      right.min[0] = middle + space
    } else {
      const [low, high] = a.min[1] < b.min[1] ? [a, b] : [b, a]
      const middle = (low.max[1] + high.min[1]) / 2
      low.max[1] = middle - space
      high.min[1] = middle + space
    }
  }

  return { kind, nodes: viewNodes, edges: routed, frames, byId, bounds: boundsOf(viewNodes, routed), representative, maxDepth: maxGroupDepth + 1 }
}

/**
 * Lane refinement in the spirit of the transpose step of layered graph drawing: swap two blocks of a
 * column, or move one into a free lane, whenever that removes crossings without bending the main path.
 * Each edge is a straight line from (rank, lane) to (rank, lane); two edges cross when their order flips
 * over the ranks they share.
 */
function untangleLanes(visible: ModelNode[], rank: Map<string, number>, lane: Map<string, number>, edges: ViewEdge[]) {
  const links = edges.filter((edge) => rank.has(edge.source) && rank.has(edge.target) && edge.type !== 'residual')
  if (links.length > 400) return
  const at = (edge: ViewEdge, r: number) => {
    const [r0, r1] = [rank.get(edge.source)!, rank.get(edge.target)!]
    const [l0, l1] = [lane.get(edge.source)!, lane.get(edge.target)!]
    return r1 === r0 ? l0 : l0 + (l1 - l0) * (r - r0) / (r1 - r0)
  }
  const crossesWith = (a: ViewEdge, b: ViewEdge) => {
    if (a.source === b.source || a.target === b.target || a.source === b.target || a.target === b.source) return false
    const from = Math.max(rank.get(a.source)!, rank.get(b.source)!)
    const to = Math.min(rank.get(a.target)!, rank.get(b.target)!)
    if (to <= from) return false
    return (at(a, from) - at(b, from)) * (at(a, to) - at(b, to)) < -1e-9
  }
  const touching = new Map<string, ViewEdge[]>()
  links.forEach((edge) => [edge.source, edge.target].forEach((id) => touching.set(id, [...(touching.get(id) ?? []), edge])))
  // Cost of the edges at some blocks: crossings dominate, then how far lines bend.
  const cost = (ids: string[]) => {
    const mine = [...new Set(ids.flatMap((id) => touching.get(id) ?? []))]
    let total = 0
    for (const edge of mine) {
      total += Math.abs(lane.get(edge.source)! - lane.get(edge.target)!) * 0.6
      for (const other of links) if (other !== edge && crossesWith(edge, other)) total += mine.includes(other) ? 5 : 10
    }
    return total
  }
  const byRank = new Map<number, ModelNode[]>()
  visible.forEach((node) => byRank.set(rank.get(node.id)!, [...(byRank.get(rank.get(node.id)!) ?? []), node]))
  for (let sweep = 0; sweep < 4; sweep++) {
    let improved = false
    for (const nodes of byRank.values()) {
      const lanes = nodes.map((node) => lane.get(node.id)!)
      const options = new Set([...lanes, Math.min(...lanes) - 1, Math.max(...lanes) + 1])
      for (const node of nodes) for (const target of options) {
        const current = lane.get(node.id)!
        if (target === current) continue
        const other = nodes.find((item) => lane.get(item.id) === target)
        // Inputs and constants stay above the main lane; the main lane keeps its blocks.
        const sourceOnly = (item: ModelNode) => !links.some((edge) => edge.target === item.id) && item.className !== 'Input'
        if (sourceOnly(node) && target > -1 || other && sourceOnly(other) && current > -1) continue
        if (current === 0 && !other) continue
        const ids = other ? [node.id, other.id] : [node.id]
        const before = cost(ids)
        lane.set(node.id, target)
        if (other) lane.set(other.id, current)
        if (cost(ids) < before - 1e-6) { improved = true; continue }
        lane.set(node.id, current)
        if (other) lane.set(other.id, target)
      }
    }
    if (!improved) break
  }
}

/** Sources whose every outgoing edge became a labelled tap. */
export function fanouts(graph: ViewGraph): { source: string; count: number; type: NonNullable<ViewEdge['type']> }[] {
  const out = new Map<string, ViewEdge[]>()
  graph.edges.forEach((edge) => out.set(edge.source, [...(out.get(edge.source) ?? []), edge]))
  return [...out].filter(([, edges]) => edges.every((edge) => edge.stub)).map(([source, edges]) => ({ source, count: edges.length, type: edges[0].type ?? 'data' }))
}

/** Far uses of a value that already feeds a nearby block become short labelled taps, so broadcasts
 *  (timestep and prompt embeddings, inputs reused by the loss) never sweep across the whole layout. */
function markStubs(kind: LayoutKind, edges: ViewEdge[], byId: Map<string, ViewNode>, owner: (id: string) => ModelNode | string): ViewEdge[] {
  const distance = (a: ViewNode, b: ViewNode) => Math.abs(a.grid[0] - b.grid[0]) + 4 * Math.abs(a.row - b.row)
  const long = (a: ViewNode, b: ViewNode) => {
    if (Math.abs(a.grid[0] - b.grid[0]) <= 1 && (kind === 'levels' ? Math.abs(a.row - b.row) <= 1 : Math.abs(a.grid[1] - b.grid[1]) <= 1.5)) return false
    // The skip connections of a U-Net are the point of the picture: they stay as bridges. Inputs reused
    // by the loss (the noise target) and anything feeding a non-spatial result become taps.
    if (kind === 'levels') return a.level !== b.level || a.node.className === 'Input' || Number.isNaN(spatialHeight(b.node))
    return a.row !== b.row || Math.abs(a.grid[0] - b.grid[0]) >= 6
  }
  // A condition broadcast to blocks two or more bands away is a tap even for its nearest use.
  const remote = (a: ViewNode, b: ViewNode) => kind === 'levels' && a.level === -1 && (b.level ?? 0) >= 1
  const bySource = new Map<string, ViewEdge[]>()
  edges.forEach((edge) => bySource.set(edge.source, [...(bySource.get(edge.source) ?? []), edge]))
  const marked = new Set<ViewEdge>()
  for (const group of bySource.values()) {
    const a = byId.get(group[0].source)
    if (!a) continue
    const ranked = group.filter((edge) => edge.type !== 'residual' && byId.has(edge.target)).sort((x, y) => distance(a, byId.get(x.target)!) - distance(a, byId.get(y.target)!))
    const first = ranked.length ? byId.get(ranked[0].target)! : undefined
    ranked.forEach((edge, index) => {
      const b = byId.get(edge.target)!
      // A sibling of the nearest use (a shortcut next to the first conv of a block) follows the same route.
      const sibling = first !== undefined && b.row === first.row && Math.abs(b.grid[0] - first.grid[0]) <= 1
      // A second use further along the same row would have to pass the first one (a condition fed to
      // every block): tap it instead of drawing a rail around.
      const passes = kind === 'flow' && first !== undefined && first !== b && a.row === b.row && first.row === b.row
        && (first.grid[0] - a.grid[0]) * (b.grid[0] - first.grid[0]) > 0
      if (index > 0 && (passes || !sibling && long(a, b)) || remote(a, b)) marked.add(edge)
    })
    // A residual that would have to follow the snake into another row (x_t reused by the sampler's
    // update) is a tap as well; within a row it stays an arc.
    if (kind === 'flow') group.forEach((edge) => { const b = byId.get(edge.target); if (edge.type === 'residual' && b && b.row !== a.row) marked.add(edge) })
  }
  const byTarget = new Map<string, ViewEdge[]>()
  edges.forEach((edge) => { if (marked.has(edge)) byTarget.set(edge.target, [...(byTarget.get(edge.target) ?? []), edge]) })
  return edges.map((edge) => {
    if (!marked.has(edge)) return edge
    const siblings = byTarget.get(edge.target)!
    return { ...edge, stub: { label: owner(edge.source), index: siblings.indexOf(edge), count: siblings.length } }
  })
}

/** Routes edges that would otherwise cut through other blocks: bridges, gutters and folds. */
function routeEdges(kind: LayoutKind, edges: ViewEdge[], nodes: ViewNode[], byId: Map<string, ViewNode>, laneHeight: number, unit: number): ViewEdge[] {
  interface Point { x: number; g: number; z?: number }
  const at = (item: ViewNode, g: number, z?: number): Point => ({ x: item.position[0], g, z })
  const route = (edge: ViewEdge, points: Point[]): ViewEdge => ({ ...edge, via: points.map((point) => [point.x, -point.g * laneHeight, point.z ?? -0.4] as Vec3) })
  // Parallel routes from different sources get their own track; routes from one source share it (a bus).
  const tracks = new Map<string, Map<string, number>>()
  const track = (rail: string, source: string) => {
    const used = tracks.get(rail) ?? new Map<string, number>()
    tracks.set(rail, used)
    if (!used.has(source)) used.set(source, used.size)
    return used.get(source)!
  }
  const segments = new Map<number, { minX: number; maxX: number; minCol: number; maxCol: number; top: number; bottom: number; dir: 1 | -1 }>()
  nodes.forEach((item) => {
    const current = segments.get(item.row) ?? { minX: Infinity, maxX: -Infinity, minCol: Infinity, maxCol: -Infinity, top: Infinity, bottom: -Infinity, dir: item.dir }
    current.minX = Math.min(current.minX, item.position[0] - item.size[0] / 2)
    current.maxX = Math.max(current.maxX, item.position[0] + item.size[0] / 2)
    current.minCol = Math.min(current.minCol, item.grid[0])
    current.maxCol = Math.max(current.maxCol, item.grid[0])
    current.top = Math.min(current.top, item.grid[1])
    current.bottom = Math.max(current.bottom, item.grid[1])
    segments.set(item.row, current)
  })
  const allMinX = Math.min(...nodes.map((item) => item.position[0] - item.size[0] / 2))
  // Would a straight line between a and b pass through another block?
  const blocked = (a: ViewNode, b: ViewNode) => {
    const [lo, hi] = [Math.min(a.grid[0], b.grid[0]), Math.max(a.grid[0], b.grid[0])]
    const [g0, g1] = [Math.min(a.grid[1], b.grid[1]) - 0.45, Math.max(a.grid[1], b.grid[1]) + 0.45]
    return nodes.some((item) => item !== a && item !== b && item.grid[0] > lo && item.grid[0] < hi && item.grid[1] >= g0 && item.grid[1] <= g1)
  }

  return edges.map((edge): ViewEdge => {
    const a = byId.get(edge.source)
    const b = byId.get(edge.target)
    if (!a || !b || edge.stub) return edge
    const near = Math.abs(a.grid[0] - b.grid[0]) <= 1
    const sa = segments.get(a.row)!
    const sb = segments.get(b.row)!

    if (kind === 'levels') {
      if (near) return edge
      if (a.level === b.level) {
        if (edge.type === 'residual' || !blocked(a, b)) return edge
        // Skip connection: a bridge just above the blocks it passes over, nested by level.
        const [lo, hi] = [Math.min(a.grid[0], b.grid[0]), Math.max(a.grid[0], b.grid[0])]
        const top = Math.min(...nodes.filter((item) => item.row === a.row && item.grid[0] >= lo && item.grid[0] <= hi).map((item) => item.grid[1]))
        const g = top - 0.62 - track(`band:${a.row}`, a.id) * 0.22
        return route(edge, [at(a, g), at(b, g)])
      }
      // Across bands: along the source's band, then straight down (or up) to the target, behind the plane.
      return (a.level ?? 0) < (b.level ?? 0) ? route(edge, [at(b, a.grid[1], -1.1)]) : route(edge, [at(a, b.grid[1], -1.1)])
    }

    if (a.row === b.row) {
      if (near || edge.type === 'residual' || !blocked(a, b)) return edge
      const above = (a.lane + b.lane) / 2 < 0
      const offset = track(`${a.row}:${above ? 'top' : 'bottom'}`, a.id) * 0.24
      return route(edge, [at(a, above ? sa.top - 0.62 - offset : sa.bottom + 0.62 + offset), at(b, above ? sa.top - 0.62 - offset : sa.bottom + 0.62 + offset)])
    }
    if (b.row === a.row + 1) {
      const right = sa.dir === 1
      const atEnd = right ? a.grid[0] >= sa.maxCol - 1 : a.grid[0] <= sa.minCol + 1
      const atStart = right ? b.grid[0] >= sb.maxCol - 1 : b.grid[0] <= sb.minCol + 1
      if (atEnd && atStart) {
        // The fold: a short bracket around the end of the row.
        const lane = track(`turn:${a.row}`, a.id)
        const x = right ? Math.max(sa.maxX, sb.maxX) + (0.9 + lane * 0.3) * unit : Math.min(sa.minX, sb.minX) - (0.9 + lane * 0.3) * unit
        return route(edge, [{ x, g: a.grid[1] }, { x, g: b.grid[1] }])
      }
      const g = sa.bottom + 0.75 + track(`gutter:${a.row}`, a.id) * 0.24
      return route(edge, [at(a, g), at(b, g)])
    }
    // Rows further apart: down the outside edge of the layout, behind the plane.
    const x = allMinX - (1.6 + track('outside', a.id) * 0.3) * unit
    const g1 = sa.bottom + 0.75 + track(`gutter:${a.row}`, a.id) * 0.24
    const g2 = sb.top - 0.75 - track(`gutter-in:${b.row}`, a.id) * 0.24
    return route(edge, [at(a, g1, -0.9), { x, g: g1, z: -0.9 }, { x, g: g2, z: -0.9 }, at(b, g2, -0.9)])
  })
}

function boundsOf(nodes: ViewNode[], edges: ViewEdge[]): ViewGraph['bounds'] {
  const points = [
    ...nodes.flatMap((node) => [
      [node.position[0] - node.size[0] / 2, node.position[1] - node.size[1] / 2, node.position[2] - node.size[2] / 2],
      [node.position[0] + node.size[0] / 2, node.position[1] + node.size[1] / 2, node.position[2] + node.size[2] / 2],
    ]),
    ...edges.flatMap((edge) => edge.via ?? []),
  ]
  if (!points.length) return { min: [0, 0, 0], max: [0, 0, 0] }
  const axis = (index: number, pick: (...values: number[]) => number) => pick(...points.map((point) => point[index]))
  return { min: [axis(0, Math.min), axis(1, Math.min), axis(2, Math.min)], max: [axis(0, Math.max), axis(1, Math.max), axis(2, Math.max)] }
}

/** Hand-authored study maps keep their positions. */
function manualGraph(model: ModelIR): ViewGraph {
  const nodes: ViewNode[] = model.nodes.map((node) => {
    const position = (node.visualization?.position ?? [0, 0, 0]) as Vec3
    const io = node.type === 'tensor' || node.type === 'latent'
    const size: Vec3 = io ? [0.42, 1.25, 1.25] : [1.8, 1.0, 0.5]
    return { id: node.id, node, position, size, grid: [position[0] / 3.35, -position[1] / 3.1], rank: Math.round(position[0] / 3.35), lane: 0, row: 0, dir: 1, collapsed: false, depth: 0, card: true }
  })
  const byId = new Map(nodes.map((node) => [node.id, node]))
  // A study map is drawn twice from this one graph, in 3D and as the flat diagram (cards on a grid):
  // lines must stay clear of the blocks in both.
  const toDiagram = (item: ViewNode) => boxOf(item, { x: item.grid[0] * CELL.w, y: item.grid[1] * CELL.h })
  const sx = 3.35 / CELL.w, sy = 3.1 / CELL.h
  const diagram = {
    sample: (edge: ViewEdge, a: ViewNode, b: ViewNode) => samplePath(diagramEdgePath({ kind: 'manual' } as ViewGraph, new Map([[a.id, toDiagram(a)], [b.id, toDiagram(b)]]), edge)).map(([x, y]) => [x * sx, -y * sy] as Point2),
    footprint: (item: ViewNode): [number, number] => { const box = toDiagram(item); return [box.w * sx, box.h * sy] },
    tolerance: 2 * sx,
  }
  const edges = clearEdges(model.edges.map((edge) => ({ ...edge })), byId, [{ sample: sceneSample, tolerance: 0.04 }, diagram], 0.3)
  return {
    kind: 'manual', nodes, edges, frames: [], byId, bounds: boundsOf(nodes, edges),
    representative: new Map(model.nodes.map((node) => [node.id, node.id])), maxDepth: 0,
  }
}

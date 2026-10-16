// Measures what the 3D scene and the flat diagram actually draw, for the integrity check.
import { checkDrawing, type IntegrityReport, type Point, type Rect } from '../../core/layout/integrity'
import type { ModelIR } from '../../core/model-ir/types'
import type { ViewGraph } from '../../core/layout/viewGraph'
import { edgeCurve } from './useViewGraph'
import { diagramBoxes, diagramEdgePath, samplePath } from './diagramGeometry'

/** The 3D scene seen from the front: glyph footprints and edge curves in the x–y plane. */
export function checkScene(model: ModelIR, graph: ViewGraph): IntegrityReport {
  const boxes = new Map<string, Rect>(graph.nodes.map((item) => [item.id, { x: item.position[0], y: item.position[1], w: item.size[0], h: item.size[1] }]))
  const lines = new Map<string, Point[]>()
  for (const edge of graph.edges) {
    const a = graph.byId.get(edge.source)
    const b = graph.byId.get(edge.target)
    if (a && b) lines.set(edge.id, edgeCurve(edge, a, b).getPoints(32).map((point) => [point.x, point.y] as Point))
  }
  return checkDrawing(model, graph, boxes, lines, 0.04)
}

/** The flat diagram: cards and SVG edge paths. */
export function checkDiagram(model: ModelIR, graph: ViewGraph): IntegrityReport {
  const boxes = diagramBoxes(graph)
  const lines = new Map<string, Point[]>()
  for (const edge of graph.edges) if (boxes.has(edge.source) && boxes.has(edge.target)) lines.set(edge.id, samplePath(diagramEdgePath(graph, boxes, edge)))
  return checkDrawing(model, graph, boxes, lines, 2)
}

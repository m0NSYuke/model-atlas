// Path tracing between two blocks: how does a value get from A to B? Follows the dataflow forward
// (or backward when B comes first) and only falls back to ignoring direction when neither exists.
// Shortcuts (residual arcs, broadcast conditions) cost more than the main data path, so the traced
// path walks through the blocks that actually transform the value.
import type { ViewEdge, ViewGraph } from '../layout/viewGraph'

const cost = (edge: ViewEdge) => edge.type === 'residual' || edge.type === 'branch' ? 6 : edge.type === 'condition' || edge.stub ? 3 : 1

export function tracePath(graph: ViewGraph, from: string, to: string): string[] | null {
  const start = graph.representative.get(from) ?? from
  const goal = graph.representative.get(to) ?? to
  if (!graph.byId.has(start) || !graph.byId.has(goal) || start === goal) return null
  const out = new Map<string, [string, number][]>()
  const into = new Map<string, [string, number][]>()
  for (const edge of graph.edges) {
    out.set(edge.source, [...(out.get(edge.source) ?? []), [edge.target, cost(edge)]])
    into.set(edge.target, [...(into.get(edge.target) ?? []), [edge.source, cost(edge)]])
  }
  const search = (next: (id: string) => [string, number][]) => {
    const distance = new Map<string, number>([[start, 0]])
    const previous = new Map<string, string>()
    const open = new Set([start])
    while (open.size) {
      let id = ''
      for (const candidate of open) if (!id || distance.get(candidate)! < distance.get(id)!) id = candidate
      open.delete(id)
      if (id === goal) break
      for (const [other, weight] of next(id)) {
        const total = distance.get(id)! + weight
        if (total < (distance.get(other) ?? Infinity)) { distance.set(other, total); previous.set(other, id); open.add(other) }
      }
    }
    if (!distance.has(goal)) return null
    const path = [goal]
    while (path[0] !== start) path.unshift(previous.get(path[0])!)
    return path
  }
  return search((id) => out.get(id) ?? []) ?? search((id) => into.get(id) ?? []) ?? search((id) => [...(out.get(id) ?? []), ...(into.get(id) ?? [])])
}

// Change comparison in the spirit of Karpathy's "change one thing, then check": what an edit did to the
// traced architecture, compared node by node with the original code.
import type { ModelIR, ModelNode, TensorShape } from '../model-ir/types'

export type DiffMark = 'added' | 'changed'

export interface ModelDiff {
  added: ModelNode[]
  removed: ModelNode[]
  reshaped: { node: ModelNode; from?: TensorShape; to?: TensorShape }[]
  params: { node: ModelNode; from: number; to: number }[]
  totalFrom: number
  totalTo: number
  /** Marks for every changed node and for the groups that contain one (so a collapsed block shows it too). */
  marks: Map<string, DiffMark>
  empty: boolean
}

const shapeKey = (shape?: TensorShape) => shape ? shape.join('×') : ''

export function diffModels(base: ModelIR, next: ModelIR): ModelDiff {
  const before = new Map(base.nodes.map((node) => [node.id, node]))
  const after = new Map(next.nodes.map((node) => [node.id, node]))
  const groups = new Set(next.nodes.map((node) => node.parentId).filter(Boolean))
  const baseGroups = new Set(base.nodes.map((node) => node.parentId).filter(Boolean))
  const added = next.nodes.filter((node) => !before.has(node.id) && node.type !== 'parameter')
  const removed = base.nodes.filter((node) => !after.has(node.id) && node.type !== 'parameter' && !(node.parentId && !after.has(node.parentId) && before.get(node.parentId)))
  const reshaped: ModelDiff['reshaped'] = []
  const params: ModelDiff['params'] = []
  for (const node of next.nodes) {
    const old = before.get(node.id)
    if (!old || node.type === 'parameter') continue
    if (shapeKey(old.outputShape) !== shapeKey(node.outputShape)) reshaped.push({ node, from: old.outputShape, to: node.outputShape })
    if (!groups.has(node.id) && !baseGroups.has(node.id) && (old.paramCount ?? 0) !== (node.paramCount ?? 0)) params.push({ node, from: old.paramCount ?? 0, to: node.paramCount ?? 0 })
  }
  const marks = new Map<string, DiffMark>()
  added.forEach((node) => marks.set(node.id, 'added'))
  for (const { node } of [...reshaped, ...params]) if (!marks.has(node.id)) marks.set(node.id, 'changed')
  const parents = new Map(next.nodes.map((node) => [node.id, node.parentId]))
  for (const id of [...marks.keys()]) for (let current = parents.get(id); current; current = parents.get(current)) if (!marks.has(current)) marks.set(current, 'changed')
  const totalFrom = base.totalParams ?? 0
  const totalTo = next.totalParams ?? 0
  return { added, removed, reshaped, params, totalFrom, totalTo, marks, empty: !added.length && !removed.length && !reshaped.length && !params.length && totalFrom === totalTo }
}

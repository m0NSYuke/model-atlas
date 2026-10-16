import type { ModelIR, ModelNode, TensorShape } from './types'

export function getNode(model: ModelIR, id: string | null): ModelNode | undefined {
  return model.nodes.find((node) => node.id === id)
}

export function formatShape(shape?: TensorShape): string {
  return shape ? `[${shape.join(', ')}]` : '—'
}

export function getChildren(model: ModelIR, parentId?: string): ModelNode[] {
  return model.nodes.filter((node) => node.parentId === parentId)
}

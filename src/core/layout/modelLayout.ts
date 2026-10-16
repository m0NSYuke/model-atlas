import type { ModelIR } from '../model-ir/types'

export function withDefaultLayout(model: ModelIR): ModelIR {
  return {
    ...model,
    nodes: model.nodes.map((node, index) => {
      if (node.visualization?.position) return node
      const row = Math.floor(index / 4)
      const column = index % 4
      const serpentineColumn = row % 2 === 0 ? column : 3 - column
      return {
        ...node,
        visualization: {
          ...node.visualization,
          position: [(serpentineColumn - 1.5) * 4.1, -row * 3.5, 0] as [number, number, number],
        },
      }
    }),
  }
}

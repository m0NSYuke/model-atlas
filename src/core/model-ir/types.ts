export type ModelNodeType =
  | 'tensor' | 'conv2d' | 'conv3d' | 'linear' | 'norm' | 'attention'
  | 'mlp' | 'encoder' | 'decoder' | 'downsample' | 'upsample'
  | 'latent' | 'sampling' | 'split' | 'concat' | 'generic'
  | 'conv1d' | 'embedding' | 'activation' | 'pool' | 'dropout' | 'add' | 'multiply'
  | 'matmul' | 'reshape' | 'recurrent' | 'loss' | 'parameter' | 'module' | 'optimizer'

export const modelNodeTypes: ModelNodeType[] = [
  'tensor', 'conv2d', 'conv3d', 'linear', 'norm', 'attention', 'mlp', 'encoder', 'decoder', 'downsample', 'upsample',
  'latent', 'sampling', 'split', 'concat', 'generic', 'conv1d', 'embedding', 'activation', 'pool', 'dropout', 'add',
  'multiply', 'matmul', 'reshape', 'recurrent', 'loss', 'parameter', 'module', 'optimizer',
]

export type TensorShape = Array<number | string>
export type Locale = 'en' | 'zh'

export interface LocalizedNodeText {
  name?: string
  description?: string
  learning?: { what?: string; why?: string; notes?: string; formula?: string }
}

export interface ModelNode {
  id: string
  name: string
  type: ModelNodeType
  parentId?: string
  description?: string
  inputShape?: TensorShape
  outputShape?: TensorShape
  parameters?: Record<string, string | number | boolean | number[]>
  source?: { file: string; startLine: number; endLine: number }
  /** Line where a traced module or operation is invoked in forward(). */
  callSite?: { file: string; line: number }
  /** PyTorch class or operation name for traced nodes, e.g. Conv2d or matmul. */
  className?: string
  /** Python variable the traced result was assigned to, e.g. `c` or `emb`. */
  variable?: string
  /** Learnable parameter count (traced models). Groups include their descendants. */
  paramCount?: number
  learning?: { what?: string; why?: string; notes?: string; formula?: string }
  localized?: Partial<Record<Locale, LocalizedNodeText>>
  visualization?: {
    position?: [number, number, number]
    scale?: [number, number, number]
    group?: string
  }
}

export interface ModelEdge {
  id: string
  source: string
  target: string
  type?: 'data' | 'residual' | 'branch' | 'condition' | 'training' | 'feedback' | 'reuse'
  // Optional routing points keep long bypasses out of the main computation lane.
  via?: [number, number, number][]
  /** Shape change on the way (view, transpose, flatten, indexing…): the target receives `shape`, not the source's output. */
  transform?: { ops: string[]; shape: TensorShape }
}

export interface ModelIR {
  id: string
  name: string
  /** 'code' when the graph is traced from Python source, 'manual' for hand-authored maps. */
  layout?: 'code' | 'manual'
  totalParams?: number
  description?: string
  localized?: Partial<Record<Locale, { name?: string; description?: string }>>
  nodes: ModelNode[]
  edges: ModelEdge[]
}

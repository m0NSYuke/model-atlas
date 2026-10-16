import { modelNodeTypes, type ModelEdge, type ModelIR, type ModelNode, type ModelNodeType, type TensorShape } from './types'
import { withDefaultLayout } from '../layout/modelLayout'

const nodeTypes = new Set<ModelNodeType>(modelNodeTypes)
const isObject = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value)
const isText = (value: unknown): value is string => typeof value === 'string' && value.trim().length > 0

function shape(value: unknown, label: string): TensorShape | undefined {
  if (value === undefined) return undefined
  if (!Array.isArray(value) || value.length > 8 || value.some((axis) => !(typeof axis === 'string' && axis.length <= 60 || typeof axis === 'number' && Number.isFinite(axis) && axis >= 0))) {
    throw new Error(`${label}: expected an array of numeric or symbolic dimensions`)
  }
  return value as TensorShape
}

function optionalText(value: unknown): string | undefined { return typeof value === 'string' ? value : undefined }

function readNode(value: unknown, index: number): ModelNode {
  if (!isObject(value) || !isText(value.id) || !isText(value.name) || !nodeTypes.has(value.type as ModelNodeType)) {
    throw new Error(`nodes[${index}]: id, name and a supported type are required`)
  }
  const visualization = value.visualization
  let position: [number, number, number] | undefined
  if (visualization !== undefined) {
    if (!isObject(visualization)) throw new Error(`nodes[${index}].visualization must be an object`)
    if (visualization.position !== undefined) {
      if (!Array.isArray(visualization.position) || visualization.position.length !== 3 || visualization.position.some((axis) => typeof axis !== 'number' || !Number.isFinite(axis) || Math.abs(axis) > 10000)) {
        throw new Error(`nodes[${index}].visualization.position must contain three finite numbers`)
      }
      position = visualization.position as [number, number, number]
    }
  }
  let source: ModelNode['source']
  if (value.source !== undefined) {
    if (!isObject(value.source) || !isText(value.source.file) || !Number.isInteger(value.source.startLine) || !Number.isInteger(value.source.endLine) || (value.source.startLine as number) < 1 || (value.source.endLine as number) < (value.source.startLine as number)) {
      throw new Error(`nodes[${index}].source needs a file and valid line range`)
    }
    source = { file: value.source.file, startLine: value.source.startLine as number, endLine: value.source.endLine as number }
  }
  let parameters: ModelNode['parameters']
  if (value.parameters !== undefined) {
    if (!isObject(value.parameters)) throw new Error(`nodes[${index}].parameters must be an object`)
    parameters = {}
    for (const [key, item] of Object.entries(value.parameters)) {
      if (typeof item === 'string' || typeof item === 'boolean' || typeof item === 'number' && Number.isFinite(item) || Array.isArray(item) && item.every((axis) => typeof axis === 'number' && Number.isFinite(axis))) parameters[key] = item
      else throw new Error(`nodes[${index}].parameters.${key} has an unsupported value`)
    }
  }
  const learning = isObject(value.learning) ? {
    what: optionalText(value.learning.what), why: optionalText(value.learning.why),
    notes: optionalText(value.learning.notes), formula: optionalText(value.learning.formula),
  } : undefined
  const localized = isObject(value.localized) ? Object.fromEntries(['en', 'zh'].flatMap((locale) => {
    const entry = value.localized as Record<string, unknown>
    const text = entry[locale]
    if (!isObject(text)) return []
    const localLearning = isObject(text.learning) ? {
      what: optionalText(text.learning.what), why: optionalText(text.learning.why),
      notes: optionalText(text.learning.notes), formula: optionalText(text.learning.formula),
    } : undefined
    return [[locale, { name: optionalText(text.name), description: optionalText(text.description), learning: localLearning }]]
  })) : undefined
  return {
    id: value.id, name: value.name, type: value.type as ModelNodeType,
    parentId: optionalText(value.parentId), description: optionalText(value.description),
    inputShape: shape(value.inputShape, `nodes[${index}].inputShape`),
    outputShape: shape(value.outputShape, `nodes[${index}].outputShape`),
    parameters, source, learning, localized,
    className: optionalText(value.className),
    paramCount: typeof value.paramCount === 'number' && Number.isFinite(value.paramCount) ? value.paramCount : undefined,
    callSite: isObject(value.callSite) && isText(value.callSite.file) && Number.isInteger(value.callSite.line) ? { file: value.callSite.file, line: value.callSite.line as number } : undefined,
    visualization: visualization ? { position, group: optionalText((visualization as Record<string, unknown>).group) } : undefined,
  }
}

export interface AtlasProject { model: ModelIR; codeFiles: Record<string, string> }

export function importModel(input: unknown): AtlasProject {
  const bundle = isObject(input) && isObject(input.model) ? input : null
  const raw = bundle ? bundle.model : input
  if (!isObject(raw) || !isText(raw.id) || !isText(raw.name) || !Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) {
    throw new Error('Expected a Model IR with id, name, nodes and edges')
  }
  if (raw.nodes.length < 1 || raw.nodes.length > 300 || raw.edges.length > 1000) throw new Error('Model must contain 1–300 nodes and at most 1000 edges')
  const nodes = raw.nodes.map(readNode)
  const ids = new Set(nodes.map((node) => node.id))
  if (ids.size !== nodes.length) throw new Error('Node ids must be unique')
  for (const node of nodes) if (node.parentId && !ids.has(node.parentId)) throw new Error(`Missing parent for ${node.id}`)
  const parents = new Map(nodes.map((node) => [node.id, node.parentId]))
  for (const node of nodes) {
    const seen = new Set<string>()
    let current: string | undefined = node.id
    while (current) {
      if (seen.has(current)) throw new Error(`Parent cycle involving ${current}`)
      seen.add(current)
      current = parents.get(current)
    }
  }
  const edges: ModelEdge[] = raw.edges.map((edge, index) => {
    if (!isObject(edge) || !isText(edge.source) || !isText(edge.target) || !ids.has(edge.source) || !ids.has(edge.target)) throw new Error(`edges[${index}] references an unknown node`)
    if (edge.type !== undefined && !['data', 'residual', 'branch', 'condition', 'training', 'feedback', 'reuse'].includes(String(edge.type))) throw new Error(`edges[${index}] has an unsupported type`)
    let via: ModelEdge['via']
    if (edge.via !== undefined) {
      if (!Array.isArray(edge.via) || edge.via.length > 12 || edge.via.some((point) => !Array.isArray(point) || point.length !== 3 || point.some((axis) => typeof axis !== 'number' || !Number.isFinite(axis) || Math.abs(axis) > 10000))) throw new Error(`edges[${index}].via must contain at most 12 finite 3D points`)
      via = edge.via as [number, number, number][]
    }
    return { id: isText(edge.id) ? edge.id : `${edge.source}-${edge.target}-${index}`, source: edge.source, target: edge.target, type: edge.type as ModelEdge['type'], via }
  })
  const codeFiles: Record<string, string> = {}
  if (bundle?.codeFiles !== undefined) {
    if (!isObject(bundle.codeFiles) || Object.keys(bundle.codeFiles).length > 20) throw new Error('codeFiles must contain at most 20 files')
    for (const [file, contents] of Object.entries(bundle.codeFiles)) {
      if (!isText(file) || file === 'model-ir.json' || file.length > 120 || typeof contents !== 'string' || contents.length > 500_000) throw new Error(`Invalid code file: ${file}`)
      codeFiles[file] = contents
    }
  }
  const localized = isObject(raw.localized) ? Object.fromEntries(['en', 'zh'].flatMap((locale) => {
    const text = raw.localized as Record<string, unknown>
    return isObject(text[locale]) ? [[locale, { name: optionalText((text[locale] as Record<string, unknown>).name), description: optionalText((text[locale] as Record<string, unknown>).description) }]] : []
  })) : undefined
  return {
    model: withDefaultLayout({ id: raw.id, name: raw.name, description: optionalText(raw.description), localized, nodes, edges, layout: raw.layout === 'code' ? 'code' : undefined, totalParams: typeof raw.totalParams === 'number' ? raw.totalParams : undefined }),
    codeFiles,
  }
}

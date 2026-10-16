// Turns Python model source into Atlas Model IR by executing it with the symbolic tracer.
import type { LocalizedNodeText, ModelIR, ModelNode } from '../model-ir/types'
import { Interpreter, type TraceNode } from './interpreter'
import { PyError } from './values'
import { describeLayer } from './layerDocs'

export interface CodeDiagnostic { message: string; file?: string; line?: number }
export interface CodeBuildResult { model: ModelIR; logs: string[] }

export class CodeBuildError extends Error {
  constructor(public diagnostic: CodeDiagnostic) { super(diagnostic.message) }
}

/** Optional `atlas.json` next to the code: the entry file to run and the project's name. */
export interface ProjectConfig { entry?: string; name?: string; zhName?: string; description?: string; zhDescription?: string }

export function projectConfig(files: Record<string, string>): ProjectConfig {
  try {
    const value = JSON.parse(files['atlas.json'] ?? 'null') as unknown
    if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
    const pick = (key: keyof ProjectConfig) => typeof (value as Record<string, unknown>)[key] === 'string' ? (value as Record<string, string>)[key] : undefined
    return { entry: pick('entry'), name: pick('name'), zhName: pick('zhName'), description: pick('description'), zhDescription: pick('zhDescription') }
  } catch { return {} }
}

/** A file that trains: it calls backward() or builds an optimizer, inside a loop. */
export function isTrainingScript(source: string): boolean {
  return /\.backward\(|\boptim\.\w+\(|\b(SGD|Adam|AdamW|RMSprop)\(/.test(source)
}

/** Local files a source imports (`from model import GPT`, `import data`), in order of appearance. */
function localImports(source: string, python: string[]): string[] {
  const names = [...source.matchAll(/^\s*(?:from\s+\.?([\w.]+)\s+import|import\s+([\w.]+))/gm)].map((match) => `${(match[1] ?? match[2]).split('.')[0]}.py`)
  return [...new Set(names)].filter((file) => python.includes(file))
}

/**
 * The order Python files run in, as `python entry.py` would: each file after the local files it imports, the
 * entry last. The entry is the one named in atlas.json, else a training script (train.py first), else
 * model.py / main.py. Other helper files run first; training scripts the entry does not import do not run.
 */
export function executionOrder(files: Record<string, string>): string[] {
  const python = Object.keys(files).filter((file) => file.endsWith('.py'))
  const chosen = projectConfig(files).entry
  const training = python.filter((file) => isTrainingScript(files[file]))
  const entry = (chosen && python.includes(chosen) ? chosen : undefined)
    ?? (training.includes('train.py') ? 'train.py' : training[0])
    ?? ['model.py', 'main.py'].find((name) => python.includes(name))
    ?? [...python].reverse().find((file) => /^\S[^\n]*\(\s*\w+/m.test(files[file]) && /^\w+\s*=\s*\w+\(/m.test(files[file]))
    ?? python[python.length - 1]
  if (!entry) return []
  const ordered: string[] = []
  const seen = new Set<string>()
  const walk = (file: string) => {
    if (seen.has(file)) return // also ends import cycles
    seen.add(file)
    localImports(files[file], python).forEach(walk)
    ordered.push(file)
  }
  walk(entry)
  const helpers = python.filter((file) => !seen.has(file) && !training.includes(file))
  return [...helpers, ...ordered]
}

export function splitDocstring(docstring?: string): { en?: string; zh?: string } {
  if (!docstring) return {}
  const en: string[] = []
  const zh: string[] = []
  docstring.split('\n').map((line) => line.trim()).forEach((line) => {
    if (/^zh\s*[:：]/i.test(line)) zh.push(line.replace(/^zh\s*[:：]\s*/i, ''))
    else if (line) en.push(line)
  })
  return { en: en.join(' ') || undefined, zh: zh.join(' ') || undefined }
}

function displayName(node: TraceNode): string {
  if (node.op) return node.name
  const last = node.name
  if (/^\d+$/.test(last)) return `${node.className} ${last}`
  return last
}

export function hasPythonModel(files: Record<string, string>): boolean {
  return Object.entries(files).some(([file, source]) => file.endsWith('.py') && /nn\.Module|torch\.nn/.test(source))
}

export function buildModelFromCode(id: string, files: Record<string, string>, given?: { name?: string; description?: string; localized?: ModelIR['localized'] }): CodeBuildResult {
  const config = projectConfig(files)
  const meta = given ?? { name: config.name, description: config.description, localized: config.zhName || config.zhDescription ? { zh: { name: config.zhName, description: config.zhDescription } } : undefined }
  const interpreter = new Interpreter()
  let trace
  try {
    trace = interpreter.run(files, executionOrder(files))
  } catch (error) {
    if (error instanceof PyError) throw new CodeBuildError({ message: error.message, file: error.file, line: error.line })
    throw new CodeBuildError({ message: error instanceof Error ? error.message : String(error) })
  }

  // Drop groups that recorded no operations (e.g. an empty nn.Sequential shortcut).
  const hasChild = new Set(trace.nodes.map((node) => node.parentId).filter(Boolean))
  const kept = trace.nodes.filter((node) => !node.group || hasChild.has(node.id))
  const keptIds = new Set(kept.map((node) => node.id))
  const resolveParent = (parentId?: string): string | undefined => {
    let current = parentId
    while (current && !keptIds.has(current)) current = trace.nodes.find((node) => node.id === current)?.parentId
    return current
  }

  const nodes: ModelNode[] = kept.map((node) => {
    const doc = splitDocstring(node.docstring)
    const generated = describeLayer(node)
    const zh: LocalizedNodeText = {
      name: generated.zh.name,
      description: doc.zh ?? generated.zh.description,
      learning: generated.zh.learning,
    }
    const parameters: ModelNode['parameters'] = { ...(node.config ?? {}) }
    if (node.inputShapes.length > 1) parameters.inputs = node.inputShapes.map((shape) => `[${shape.join(', ')}]`).join(' + ')
    if (node.outputShapes.length > 1) parameters.outputs = node.outputShapes.length
    return {
      id: node.id,
      name: displayName(node),
      type: node.type,
      parentId: resolveParent(node.parentId),
      className: node.className,
      variable: node.variable,
      description: doc.en ?? generated.en.description,
      inputShape: node.inputShapes[0],
      outputShape: node.outputShapes[0],
      parameters: Object.keys(parameters).length ? parameters : undefined,
      paramCount: node.params,
      source: node.source && node.source.file in files ? { file: node.source.file, startLine: node.source.line, endLine: Math.max(node.source.line, node.source.endLine) } : undefined,
      callSite: node.callSite && node.callSite.file in files && node.callSite.line !== node.source?.line ? { file: node.callSite.file, line: node.callSite.line } : undefined,
      learning: generated.en.learning,
      localized: { zh },
    }
  })

  const rootDoc = splitDocstring(trace.rootDoc)
  return {
    logs: trace.logs,
    model: {
      id,
      name: meta.name ?? trace.rootClass,
      description: meta.description ?? rootDoc.en ?? `${trace.rootClass} traced from Python source`,
      localized: meta.localized ?? (rootDoc.zh ? { zh: { description: rootDoc.zh } } : undefined),
      layout: 'code',
      totalParams: trace.totalParams,
      nodes,
      edges: trace.edges.filter((edge) => keptIds.has(edge.source) && keptIds.has(edge.target)),
    },
  }
}

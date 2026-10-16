// A small Python interpreter that executes PyTorch model code symbolically.
// Tensors carry shapes instead of data; calling a model records a graph of modules and operations.
import type { ModelEdge, ModelNodeType } from '../model-ir/types'
import { PySyntaxError } from './lexer'
import { parsePython, type Expr, type Param, type Stmt } from './parser'
import { createLayer, interpolateShape, isLayer, layerType, lossShape, nTuple, supportedLayers, toInt, toInts, toNumber } from './layers'
import * as S from './shapes'
import { PRELUDE_FILE, preludeSource } from './prelude'
import { TorchSize, convert, formatSpec, percentFormat, pyCompare, pyRepr, pyStr, strFormat } from './format'
import {
  BoundMethod, ContextManager, Dtype, Ellipsis, ModuleObj, NN_MODULE, Namespace, NativeFn, PyClass, PyDict, PyError,
  PyFunction, PyInstance, PyList, PyRange, PySet, PySlice, PyTuple, Scope, SuperProxy, Tensor, formatValue, typeName,
  type NativeImpl, type Primitive, type Site, type Value,
} from './values'

export interface TraceNode {
  id: string
  name: string
  className: string
  type: ModelNodeType
  parentId?: string
  inputShapes: number[][]
  outputShapes: number[][]
  config?: Record<string, string | number | boolean | number[]>
  params: number
  source?: Site
  callSite?: Site
  docstring?: string
  order: number
  group: boolean
  op?: string
  /** Variable the result was first assigned to in forward(), e.g. `c` in `c = t_emb + y_emb`. */
  variable?: string
}

export interface TraceResult {
  rootClass: string
  rootDoc?: string
  rootSite?: Site
  nodes: TraceNode[]
  edges: ModelEdge[]
  totalParams: number
  logs: string[]
}

class ReturnSignal { constructor(public value: Value) {} }
class BreakSignal {}
class ContinueSignal {}

const MAX_STEPS = 400_000
const MAX_NODES = 900

const noop = (name: string) => new NativeFn(name, () => null)

function tupleOf(values: Value[]): PyTuple { return new PyTuple(values) }

function isTruthy(value: Value): boolean {
  if (value === null || value === false || value === 0 || value === '') return false
  if (value instanceof PyList || value instanceof PyTuple) return value.items.length > 0
  if (value instanceof PyDict) return value.map.size > 0
  if (value instanceof PyRange) return value.items.length > 0
  if (value instanceof ModuleObj && (value.kind === 'modulelist' || value.kind === 'sequential' || value.kind === 'moduledict')) return value.children.size > 0
  return true
}

function equals(a: Value, b: Value): boolean {
  if (a === b) return true
  if ((a instanceof PyTuple || a instanceof PyList) && (b instanceof PyTuple || b instanceof PyList)) {
    return (a instanceof PyTuple) === (b instanceof PyTuple) && a.items.length === b.items.length && a.items.every((item, index) => equals(item, b.items[index]))
  }
  if (typeof a === 'boolean' && typeof b === 'number') return Number(a) === b
  if (typeof a === 'number' && typeof b === 'boolean') return a === Number(b)
  return false
}

function shapeArgs(args: Value[], what: string): number[] {
  if (args.length === 1 && (args[0] instanceof PyTuple || args[0] instanceof PyList)) return toInts(args[0], what)
  return args.map((arg, index) => toInt(arg, `${what}[${index}]`))
}

function tensorsIn(value: Value): Tensor[] {
  if (value instanceof Tensor) return [value]
  if (value instanceof PyTuple || value instanceof PyList) return value.items.flatMap(tensorsIn)
  if (value instanceof PyDict) return [...value.map.values()].flatMap(tensorsIn)
  return []
}

function nestedListShape(value: Value): number[] {
  if (value instanceof PyList || value instanceof PyTuple) return [value.items.length, ...(value.items.length ? nestedListShape(value.items[0]) : [])]
  return []
}

class Tracer {
  nodes: TraceNode[] = []
  edges: ModelEdge[] = []
  byId = new Map<string, TraceNode>()
  scope: { id?: string }[] = [{}]
  finished = false
  private edgeKeys = new Set<string>()
  private predecessors = new Map<string, string[]>()

  /** True when `ancestor` feeds into `id` through recorded edges. */
  isAncestor(ancestor: string, id: string): boolean {
    const stack = [id]
    const seen = new Set<string>()
    while (stack.length && seen.size < 4000) {
      const current = stack.pop()!
      for (const previous of this.predecessors.get(current) ?? []) {
        if (previous === ancestor) return true
        if (!seen.has(previous)) { seen.add(previous); stack.push(previous) }
      }
    }
    return false
  }

  uniqueId(base: string): string {
    const clean = base || 'node'
    if (!this.byId.has(clean)) return clean
    let index = 2
    while (this.byId.has(`${clean}_${index}`)) index++
    return `${clean}_${index}`
  }

  get parentId(): string | undefined { return this.scope[this.scope.length - 1].id }

  add(node: Omit<TraceNode, 'order' | 'parentId'> & { parentId?: string }): TraceNode {
    if (this.nodes.length >= MAX_NODES) throw new PyError(`The traced graph has more than ${MAX_NODES} operations. Reduce depth or layer count to visualize it.`)
    // An explicit `parentId: undefined` places the node at the top level (model inputs and outputs).
    const full: TraceNode = { ...node, parentId: 'parentId' in node ? node.parentId : this.parentId, order: this.nodes.length }
    this.nodes.push(full)
    this.byId.set(full.id, full)
    return full
  }

  connect(source: string, target: string, type: ModelEdge['type'] = 'data', tensor?: Tensor): void {
    if (source === target) return
    const key = `${source}>${target}`
    if (this.edgeKeys.has(key)) return
    this.edgeKeys.add(key)
    // Record a shape change between producer and consumer so the drawn interface matches what flows.
    const outputs = this.byId.get(source)?.outputShapes ?? []
    const changed = tensor && !outputs.some((shape) => shape.length === tensor.shape.length && shape.every((size, index) => size === tensor.shape[index]))
    this.edges.push(changed ? { id: key, source, target, type, transform: { ops: [...new Set(tensor.reshapes ?? ['reshape'])], shape: [...tensor.shape] } } : { id: key, source, target, type })
    this.predecessors.set(target, [...(this.predecessors.get(target) ?? []), source])
  }

  isComputation(id: string): boolean {
    const node = this.byId.get(id)
    return Boolean(node && node.type !== 'parameter' && node.type !== 'tensor')
  }

  scopedId(op: string): string {
    const parent = this.parentId
    return this.uniqueId(parent ? `${parent}/${op}` : op)
  }
}

const NO_MODEL = 'No model was run: the code never calls an nn.Module. Create the model and call it once on an example input, e.g. model(torch.randn(1, 3, 32, 32)), inside `if __name__ == "__main__":`.'

interface Frame { owner?: PyClass; self?: Value }

export class Interpreter {
  private globals = new Scope()
  private builtins = new Scope()
  private steps = 0
  private depth = 0
  private frames: Frame[] = []
  private tracer: Tracer | null = null
  private traceDepth = 0
  private location: Site = { file: '', line: 0, endLine: 0 }
  private sources: Record<string, string> = {}
  private rootModule: ModuleObj | null = null
  // Training: once an optimizer exists or a DataLoader yields a batch, the trace covers one whole training
  // step (batch → model → loss → backward → optimizer.step) and every loop around it runs only once.
  private training: 'off' | 'open' | 'done' = 'off'
  private gradient: Tensor | null = null
  private stepNode: TraceNode | null = null
  private loopSizes: number[] = []
  private lastLoaderSize: number | null = null
  private batchNodes = new Set<string>()
  private rootGroup: TraceNode | undefined
  /** Loading a sample inside the DataLoader is data preparation, not part of the drawn step. */
  private muted = 0
  logs: string[] = []

  /** Script mode: a file that never calls an nn.Module is traced from its first line, op by op. */
  constructor(private readonly script = false) {
    this.globals = new Scope(this.builtins)
    this.installBuiltins()
    this.installPrelude()
  }

  /** Defines nn.TransformerEncoder, einops layers, checkpoint … from the Python prelude. */
  private installPrelude(): void {
    preludeBody ??= parsePython(preludeSource, PRELUDE_FILE)
    const scope = new Scope(this.builtins)
    this.execBlock(preludeBody, scope, PRELUDE_FILE)
    const into = (path: string, names: string[]) => {
      const target = this.resolveModule(path)
      if (target instanceof Namespace) names.forEach((name) => target.attrs.set(name, scope.vars.get(name)!))
    }
    into('torch.nn', ['TransformerEncoder', 'TransformerDecoder', 'Transformer', 'ParameterList', 'ParameterDict'])
    const einops = this.resolveModule('einops')
    if (einops instanceof Namespace) einops.attrs.set('layers', new Namespace('einops.layers', new Map([['torch', new Namespace('einops.layers.torch', new Map())]])))
    into('einops.layers.torch', ['Rearrange', 'Reduce'])
    const utils = this.resolveModule('torch.utils')
    if (utils instanceof Namespace) utils.attrs.set('checkpoint', new Namespace('torch.utils.checkpoint', new Map()))
    into('torch.utils.checkpoint', ['checkpoint', 'checkpoint_sequential'])
  }

  // ───────────────────────────── entry point ─────────────────────────────

  run(files: Record<string, string>, order: string[]): TraceResult {
    this.sources = files
    const programs = order.map((file) => {
      try {
        return { file, body: parsePython(files[file], file) }
      } catch (error) {
        if (error instanceof PySyntaxError) throw new PyError(`SyntaxError: ${error.message}`, error.line, error.file ?? file)
        throw error
      }
    })
    if (this.script) { this.tracer = new Tracer(); this.traceDepth = 1 }
    for (const program of programs) {
      // Only the entry file runs its `if __name__ == "__main__":` block, as with `python entry.py`.
      this.globals.vars.set('__name__', program === programs[programs.length - 1] ? '__main__' : program.file.replace(/\.py$/, ''))
      this.execBlock(program.body, this.globals, program.file)
    }
    if (this.training === 'open') this.finishTrace()
    if (this.script && this.tracer && !this.rootModule) {
      if (!this.tracer.nodes.some((node) => node.op)) throw new PyError(NO_MODEL)
      this.traceDepth = 0
      this.finishTrace()
      return { rootClass: (order[order.length - 1] ?? 'script').replace(/\.py$/, ''), nodes: this.tracer.nodes, edges: this.tracer.edges, totalParams: 0, logs: this.logs }
    }
    // Plain tensor code with no model: run it again with every top-level tensor operation drawn.
    if (!this.rootModule && this.training === 'off' && !this.script) return new Interpreter(true).run(files, order)
    if (!this.tracer || !this.rootModule) throw new PyError(NO_MODEL)
    const total = this.rootModule ? this.rootModule.parameters().reduce((sum, tensor) => sum + tensor.numel, 0) : 0
    return {
      rootClass: this.rootModule?.className ?? 'Model',
      rootDoc: this.rootModule?.cls?.docstring,
      rootSite: this.rootModule?.site,
      nodes: this.tracer.nodes,
      edges: this.tracer.edges,
      totalParams: total,
      logs: this.logs,
    }
  }

  private get recording(): boolean { return Boolean(this.tracer && !this.tracer.finished && !this.muted && (this.traceDepth > 0 || this.training === 'open')) }

  /** Starts tracing a training step. A forward pass traced earlier (an example call in model.py) is replaced. */
  private openTraining(): void {
    if (this.training !== 'off') return
    this.training = 'open'
    if (this.tracer?.finished) {
      const reset = (module: ModuleObj) => { module.callCount = 0; module.children.forEach(reset) }
      if (this.rootModule) reset(this.rootModule)
      this.tracer = null
      this.rootModule = null
    }
    this.tracer ??= new Tracer()
  }

  /** Closes the trace: groups add up their descendants' parameters. */
  private finishTrace(): void {
    const tracer = this.tracer
    if (!tracer || tracer.finished) return
    tracer.finished = true
    if (this.training === 'open') this.training = 'done'
    const children = new Map<string, TraceNode[]>()
    tracer.nodes.forEach((node) => { if (node.parentId) children.set(node.parentId, [...(children.get(node.parentId) ?? []), node]) })
    const total = (node: TraceNode): number => node.group ? (children.get(node.id) ?? []).reduce((sum, child) => sum + total(child), 0) : node.params
    tracer.nodes.forEach((node) => { if (node.group) node.params = total(node) })
    // Shared weights (a tied embedding and output head) count once for the whole model, as in PyTorch.
    if (this.rootGroup && this.rootModule) this.rootGroup.params = this.rootModule.parameters().reduce((sum, tensor) => sum + tensor.numel, 0)
  }

  /** A loop whose body completed the traced training step runs only that once; its size is noted on the step. */
  private cutLoop(size: number): void {
    this.loopSizes.push(size)
    if (!this.stepNode) return
    const sizes = [...this.loopSizes].reverse()
    const steps = sizes.reduce((product, value) => product * value, 1)
    this.stepNode.config = { ...this.stepNode.config, loop: sizes.join(' × '), steps }
  }

  private error(message: string): PyError { return new PyError(message) }

  // ───────────────────────────── statements ─────────────────────────────

  private execBlock(body: Stmt[], scope: Scope, file: string): void {
    for (const statement of body) this.exec(statement, scope, file)
  }

  private exec(statement: Stmt, scope: Scope, file: string): void {
    if (++this.steps > MAX_STEPS) throw new PyError('Execution limit reached. Check for very large loops.', statement.line, file)
    // Prelude code keeps the user's line as the location, so its operations and errors point at the user's call.
    const internal = file === PRELUDE_FILE
    const previous = this.location
    if (!internal) this.location = { file, line: statement.line, endLine: statement.endLine }
    try {
      this.execInner(statement, scope, file)
    } catch (error) {
      if (error instanceof ReturnSignal || error instanceof BreakSignal || error instanceof ContinueSignal) throw error
      if (error instanceof PyError) {
        if (error.line === undefined && !internal) { error.line = statement.line; error.file = file }
        if (error.file === PRELUDE_FILE) { error.line = undefined; error.file = undefined }
        throw error
      }
      if (internal) throw error
      if (error instanceof RangeError) throw new PyError('Recursion is too deep', statement.line, file)
      throw new PyError(error instanceof Error ? `Internal tracer error: ${error.message}` : 'Internal tracer error', statement.line, file)
    } finally {
      this.location = previous
    }
  }

  private execInner(statement: Stmt, scope: Scope, file: string): void {
    switch (statement.kind) {
      case 'expr': this.evaluate(statement.value, scope, file); return
      case 'assign': {
        const value = this.evaluate(statement.value, scope, file)
        for (const target of statement.targets) this.assign(target, value, scope, file)
        return
      }
      case 'annassign': if (statement.value) this.assign(statement.target, this.evaluate(statement.value, scope, file), scope, file); return
      case 'augassign': {
        const current = this.evaluate(statement.target, scope, file)
        const value = this.evaluate(statement.value, scope, file)
        if (current instanceof PyList && statement.op === '+') { current.items.push(...this.iterate(value)); return }
        this.assign(statement.target, this.binary(statement.op, current, value, statement.line), scope, file)
        return
      }
      case 'return': throw new ReturnSignal(statement.value ? this.evaluate(statement.value, scope, file) : null)
      case 'pass': case 'global': case 'del': return
      case 'break': throw new BreakSignal()
      case 'continue': throw new ContinueSignal()
      case 'import': this.importNames(statement.names, scope); return
      case 'if':
        if (isTruthy(this.evaluate(statement.test, scope, file))) this.execBlock(statement.body, scope, file)
        else this.execBlock(statement.orelse, scope, file)
        return
      case 'for': {
        this.lastLoaderSize = null
        const items = this.iterate(this.evaluate(statement.iter, scope, file))
        const size = this.lastLoaderSize ?? items.length
        const training = this.training
        let broke = false
        for (const item of items) {
          this.assign(statement.target, item, scope, file)
          this.nameBatch(statement.target, scope)
          try { this.execBlock(statement.body, scope, file) } catch (signal) {
            if (signal instanceof BreakSignal) { broke = true; break }
            if (signal instanceof ContinueSignal) { if (training !== 'done' && this.training === 'done') { this.cutLoop(size); break } continue }
            throw signal
          }
          if (training !== 'done' && this.training === 'done') { this.cutLoop(size); break }
        }
        if (!broke) this.execBlock(statement.orelse, scope, file)
        return
      }
      case 'while': {
        let guard = 0
        const training = this.training
        while (isTruthy(this.evaluate(statement.test, scope, file))) {
          if (++guard > 10_000) throw this.error('while loop exceeded 10,000 iterations')
          try { this.execBlock(statement.body, scope, file) } catch (signal) {
            if (signal instanceof BreakSignal) break
            if (!(signal instanceof ContinueSignal)) throw signal
          }
          // A training loop's length is unknown here; one traced step stands for all of them.
          if (training !== 'done' && this.training === 'done') break
        }
        return
      }
      case 'with':
        for (const item of statement.items) {
          const value = this.evaluate(item.value, scope, file)
          if (item.target) this.assign(item.target, value, scope, file)
        }
        this.execBlock(statement.body, scope, file)
        return
      case 'try':
        try { this.execBlock(statement.body, scope, file) } catch (error) {
          if (!(error instanceof PyError) || !statement.handlers.length) throw error
          this.execBlock(statement.handlers[0], scope, file)
        } finally { this.execBlock(statement.finalbody, scope, file) }
        return
      case 'raise': {
        const value = statement.value ? this.evaluate(statement.value, scope, file) : null
        throw this.error(typeof value === 'string' ? value : value instanceof PyInstance ? String(value.attrs.get('message') ?? value.cls.name) : `Exception raised: ${formatValue(value)}`)
      }
      case 'assert': {
        if (!isTruthy(this.evaluate(statement.test, scope, file))) {
          if (statement.message) throw this.error(`AssertionError: ${formatValue(this.evaluate(statement.message, scope, file))}`)
          const text = (this.sources[file]?.split('\n')[statement.line - 1] ?? '').trim()
          const test = statement.test
          // Show what the left side of a failed comparison evaluated to, e.g. 128 % 5 → 3.
          const detail = test.kind === 'compare' ? ` (left side = ${formatValue(this.evaluate(test.left, scope, file))})` : ''
          throw this.error(`AssertionError: ${text || 'assert failed'}${detail}`)
        }
        return
      }
      case 'def': scope.vars.set(statement.name, this.defineFunction(statement, scope, file)); return
      case 'class': this.defineClass(statement, scope, file); return
    }
  }

  private defineFunction(statement: Extract<Stmt, { kind: 'def' }>, scope: Scope, file: string): PyFunction {
    const fn = this.makeFunction(statement.name, statement.params, statement.body, scope, file, statement.line, statement.endLine, statement.docstring)
    for (const decorator of statement.decorators) {
      const name = decorator.kind === 'name' ? decorator.id : decorator.kind === 'attr' ? decorator.name : decorator.kind === 'call' && decorator.func.kind === 'name' ? decorator.func.id : ''
      if (name === 'staticmethod') fn.isStatic = true
      if (name === 'classmethod') fn.isClassMethod = true
      if (name === 'property') fn.isProperty = true
    }
    return fn
  }

  private makeFunction(name: string, params: Param[], body: Stmt[] | null, scope: Scope, file: string, line: number, endLine: number, docstring?: string, expressionBody?: Expr): PyFunction {
    const defaults = new Map<string, Value>()
    for (const param of params) if (param.defaultValue) defaults.set(param.name, this.evaluate(param.defaultValue, scope, file))
    return new PyFunction(name, params, body, scope, defaults, { file, line, endLine }, docstring, expressionBody)
  }

  private defineClass(statement: Extract<Stmt, { kind: 'class' }>, scope: Scope, file: string): void {
    const bases = statement.bases.map((base) => this.evaluate(base, scope, file))
    const cls = new PyClass(statement.name, bases, { file, line: statement.line, endLine: statement.endLine }, statement.docstring)
    const classScope = new Scope(scope)
    for (const item of statement.body) {
      if (item.kind === 'def') {
        // Methods close over the enclosing scope, not the class body.
        classScope.vars.set(item.name, this.defineFunction(item, scope, file))
        continue
      }
      if (item.kind === 'annassign' && item.target.kind === 'name') {
        const value = item.value ? this.evaluate(item.value, classScope, file) : undefined
        let factory: Value | undefined
        let defaultValue = value
        if (value instanceof PyDict && value.map.has('__field__')) { factory = value.map.get('default_factory') ?? undefined; defaultValue = value.map.get('default') }
        cls.fields.push({ name: item.target.id, defaultValue, factory })
        if (defaultValue !== undefined) classScope.vars.set(item.target.id, defaultValue)
        continue
      }
      this.exec(item, classScope, file)
    }
    classScope.vars.forEach((value, key) => cls.attrs.set(key, value))
    cls.isDataclass = statement.decorators.some((decorator) => (decorator.kind === 'name' && decorator.id === 'dataclass') || (decorator.kind === 'call' && decorator.func.kind === 'name' && decorator.func.id === 'dataclass') || (decorator.kind === 'attr' && decorator.name === 'dataclass'))
    scope.vars.set(statement.name, cls)
  }

  private importNames(names: { module: string; name?: string; asName: string }[], scope: Scope): void {
    for (const item of names) {
      const module = this.resolveModule(item.module)
      if (!module) continue // local project modules share one namespace
      if (!item.name) { scope.vars.set(item.asName, module); continue }
      if (item.name === '*') { if (module instanceof Namespace) module.attrs.forEach((value, key) => scope.vars.set(key, value)); continue }
      if (module instanceof Namespace && module.attrs.has(item.name)) scope.vars.set(item.asName, module.attrs.get(item.name)!)
      else if (module instanceof Namespace && module.permissive) scope.vars.set(item.asName, null)
    }
  }

  private resolveModule(path: string): Value | null {
    const parts = path.split('.')
    let current: Value | undefined = this.builtins.vars.get(`__module_${parts[0]}`)
    for (const part of parts.slice(1)) {
      if (!(current instanceof Namespace)) return null
      current = current.attrs.get(part)
    }
    return current ?? null
  }

  // ───────────────────────────── assignment ─────────────────────────────

  private assign(target: Expr, value: Value, scope: Scope, file: string): void {
    switch (target.kind) {
      case 'name': {
        scope.vars.set(target.id, value)
        const producer = value instanceof Tensor && value.producer ? this.tracer?.byId.get(value.producer) : undefined
        if (producer && !producer.variable && this.recording) producer.variable = target.id
        return
      }
      case 'attr': this.setAttr(this.evaluate(target.object, scope, file), target.name, value); return
      case 'subscript': {
        const object = this.evaluate(target.object, scope, file)
        const index = this.evaluate(target.index, scope, file)
        if (object instanceof PyList) { object.items[this.listIndex(index, object.items.length)] = value; return }
        if (object instanceof PyDict) { object.map.set(this.key(index), value); return }
        if (object instanceof Tensor) return // in-place writes do not change shapes
        if (object instanceof ModuleObj && object.kind === 'moduledict' && value instanceof ModuleObj) { object.addChild(String(index), value); return }
        const setItem = object instanceof ModuleObj && object.kind === 'user' ? this.bindFrom(object.cls!, '__setitem__', object) : object instanceof PyInstance ? this.bindFrom(object.cls, '__setitem__', object) : undefined
        if (setItem !== undefined) { this.call(setItem, [index, value], new Map(), this.location); return }
        throw this.error(`'${typeName(object)}' does not support item assignment`)
      }
      case 'tuple': case 'list': {
        const values = this.iterate(value)
        const starIndex = target.items.findIndex((item) => item.kind === 'starred')
        if (starIndex < 0 && values.length !== target.items.length) throw this.error(`Cannot unpack ${values.length} values into ${target.items.length} names`)
        if (starIndex < 0) { target.items.forEach((item, index) => this.assign(item, values[index], scope, file)); return }
        const after = target.items.length - starIndex - 1
        target.items.slice(0, starIndex).forEach((item, index) => this.assign(item, values[index], scope, file))
        const starred = target.items[starIndex]
        if (starred.kind === 'starred') this.assign(starred.value, new PyList(values.slice(starIndex, values.length - after)), scope, file)
        target.items.slice(starIndex + 1).forEach((item, index) => this.assign(item, values[values.length - after + index], scope, file))
        return
      }
      case 'starred': this.assign(target.value, value, scope, file); return
      default: throw this.error('Unsupported assignment target')
    }
  }

  // ───────────────────────────── expressions ─────────────────────────────

  private evaluate(expr: Expr, scope: Scope, file: string): Value {
    switch (expr.kind) {
      case 'const': return expr.value
      case 'ellipsis': return Ellipsis
      case 'name': {
        const value = scope.lookup(expr.id)
        if (value === undefined) throw new PyError(`NameError: name '${expr.id}' is not defined`, expr.line, file)
        return value
      }
      case 'attr': return this.getAttr(this.evaluate(expr.object, scope, file), expr.name)
      case 'subscript': return this.getItem(this.evaluate(expr.object, scope, file), this.evaluate(expr.index, scope, file))
      case 'slice': {
        const part = (item?: Expr) => { if (!item) return null; const value = this.evaluate(item, scope, file); return value === null ? null : toInt(value, 'slice') }
        return new PySlice(part(expr.lower), part(expr.upper), part(expr.step))
      }
      case 'call': {
        const fn = this.evaluate(expr.func, scope, file)
        const args: Value[] = []
        const kwargs = new Map<string, Value>()
        for (const arg of expr.args) {
          const value = this.evaluate(arg.value, scope, file)
          if (arg.star === '*') args.push(...this.iterate(value))
          else if (arg.star === '**') { if (value instanceof PyDict) value.map.forEach((item, key) => kwargs.set(String(key), item)) }
          else if (arg.name) kwargs.set(arg.name, value)
          else args.push(value)
        }
        return this.call(fn, args, kwargs, { file, line: expr.line, endLine: expr.endLine })
      }
      case 'binop': return this.binary(expr.op, this.evaluate(expr.left, scope, file), this.evaluate(expr.right, scope, file), expr.line)
      case 'unary': {
        const value = this.evaluate(expr.operand, scope, file)
        if (expr.op === 'not') return !isTruthy(value)
        if (value instanceof Tensor) return this.passThrough(value, value.shape, value.dtype)
        if (expr.op === '-') return -toNumber(value, 'operand')
        if (expr.op === '~') return ~toInt(value, 'operand')
        return toNumber(value, 'operand')
      }
      case 'bool': {
        let value: Value = null
        for (const item of expr.values) {
          value = this.evaluate(item, scope, file)
          if (expr.op === 'and' ? !isTruthy(value) : isTruthy(value)) return value
        }
        return value
      }
      case 'compare': {
        let left = this.evaluate(expr.left, scope, file)
        for (let index = 0; index < expr.ops.length; index++) {
          const right = this.evaluate(expr.comparators[index], scope, file)
          const result = this.compare(expr.ops[index], left, right)
          if (result instanceof Tensor) return result
          if (!result) return false
          left = right
        }
        return true
      }
      case 'ifexp': return isTruthy(this.evaluate(expr.test, scope, file)) ? this.evaluate(expr.body, scope, file) : this.evaluate(expr.orelse, scope, file)
      case 'tuple': return new PyTuple(this.evaluateItems(expr.items, scope, file))
      case 'list': return new PyList(this.evaluateItems(expr.items, scope, file))
      case 'set': return new PySet(this.evaluateItems(expr.items, scope, file))
      case 'dict': {
        const dict = new PyDict()
        expr.keys.forEach((key, index) => {
          const value = this.evaluate(expr.values[index], scope, file)
          if (key === null) { if (value instanceof PyDict) value.map.forEach((item, name) => dict.map.set(name, item)) }
          else dict.map.set(this.key(this.evaluate(key, scope, file)), value)
        })
        return dict
      }
      case 'comp': {
        const results: Value[] = []
        const entries: [Primitive, Value][] = []
        const run = (comp: Extract<Expr, { kind: 'comp' }>, outer: Scope) => {
          for (const item of this.iterate(this.evaluate(comp.iter, outer, file))) {
            const inner = new Scope(outer)
            this.assign(comp.target, item, inner, file)
            if (!comp.conditions.every((condition) => isTruthy(this.evaluate(condition, inner, file)))) continue
            if (comp.element.kind === 'comp' && comp.element.nested) { run(comp.element, inner); continue }
            if (comp.type === 'dict') entries.push([this.key(this.evaluate(comp.element, inner, file)), this.evaluate(comp.value!, inner, file)])
            else results.push(this.evaluate(comp.element, inner, file))
          }
        }
        run(expr, scope)
        if (expr.type === 'dict') return new PyDict(entries)
        return expr.type === 'set' ? new PySet(results) : new PyList(results)
      }
      case 'lambda': return this.makeFunction('<lambda>', expr.params, null, scope, file, expr.line, expr.line, undefined, expr.body)
      case 'starred': return this.evaluate(expr.value, scope, file)
      case 'namedexpr': {
        const value = this.evaluate(expr.value, scope, file)
        this.assign({ kind: 'name', id: expr.target, line: expr.line }, value, scope, file)
        return value
      }
      case 'fstring': return this.formatParts(expr.parts, scope, file)
    }
  }

  private formatParts(parts: Extract<Expr, { kind: 'fstring' }>['parts'], scope: Scope, file: string): string {
    return parts.map((part) => {
      if (typeof part === 'string') return part
      const value = this.evaluate(part.expr, scope, file)
      const spec = part.spec ? this.formatParts(part.spec, scope, file) : ''
      const text = this.pyFormat(convert(value, part.conversion ?? (part.debug && !part.spec ? 'r' : undefined)), spec)
      return part.debug ? part.debug + text : text
    }).join('')
  }

  private pyFormat(value: Value, spec: string): string {
    try { return formatSpec(value, spec) } catch (error) { throw this.error(`ValueError: ${(error as Error).message}`) }
  }

  /** Follows `.attr` and `[index]` steps of a str.format field such as `{cfg.dim}` or `{shape[0]}`. */
  private formatLookup(value: Value, path: string): Value {
    let current = value
    for (const [, attr, index] of path.matchAll(/\.([A-Za-z_]\w*)|\[([^\]]*)\]/g)) {
      current = attr !== undefined ? this.getAttr(current, attr) : this.getItem(current, /^-?\d+$/.test(index) ? Number(index) : index)
    }
    return current
  }

  private evaluateItems(items: Expr[], scope: Scope, file: string): Value[] {
    const values: Value[] = []
    for (const item of items) {
      if (item.kind === 'starred') values.push(...this.iterate(this.evaluate(item.value, scope, file)))
      else values.push(this.evaluate(item, scope, file))
    }
    return values
  }

  /** copy.deepcopy: modules get fresh parameter tensors of the same shapes, as PyTorch clones do. */
  private deepCopy(value: Value, memo: Map<object, Value>): Value {
    if (value === null || typeof value !== 'object') return value
    if (memo.has(value)) return memo.get(value)!
    if (value instanceof Tensor) {
      const copy = new Tensor([...value.shape], null, value.dtype, value.role)
      copy.label = value.label
      memo.set(value, copy)
      return copy
    }
    if (value instanceof PyList || value instanceof PyTuple) {
      const copy = value instanceof PyList ? new PyList([]) : new (value instanceof TorchSize ? TorchSize : PyTuple)([])
      memo.set(value, copy)
      copy.items.push(...value.items.map((item) => this.deepCopy(item, memo)))
      return copy
    }
    if (value instanceof PyDict) {
      const copy = new PyDict()
      memo.set(value, copy)
      value.map.forEach((item, key) => copy.map.set(key, this.deepCopy(item, memo)))
      return copy
    }
    if (value instanceof ModuleObj) {
      const layer = value.layer && {
        ...value.layer,
        params: new Map([...value.layer.params].map(([key, tensor]) => [key, this.deepCopy(tensor, memo) as Tensor])),
        props: new Map([...value.layer.props].map(([key, item]) => [key, item instanceof Tensor ? this.deepCopy(item, memo) : item])),
      }
      const copy = new ModuleObj(value.kind, value.className, value.site, value.cls, layer)
      memo.set(value, copy)
      value.attrs.forEach((item, key) => {
        const cloned = this.deepCopy(item, memo)
        if (cloned instanceof Tensor && cloned.role !== 'value') cloned.owner = copy
        copy.attrs.set(key, cloned)
      })
      value.children.forEach((child, key) => copy.addChild(key, this.deepCopy(child, memo) as ModuleObj))
      return copy
    }
    if (value instanceof PyInstance) {
      const copy = new PyInstance(value.cls)
      memo.set(value, copy)
      value.attrs.forEach((item, key) => copy.attrs.set(key, this.deepCopy(item, memo)))
      return copy
    }
    return value
  }

  private shallowCopy(value: Value): Value {
    if (value instanceof PyList) return new PyList([...value.items])
    if (value instanceof PyDict) return new PyDict([...value.map.entries()])
    if (value instanceof PyInstance) { const copy = new PyInstance(value.cls); value.attrs.forEach((item, key) => copy.attrs.set(key, item)); return copy }
    return value
  }

  private compareValues(a: Value, b: Value): number {
    try { return pyCompare(a, b) } catch (error) { throw this.error(`TypeError: ${(error as Error).message}`) }
  }

  private key(value: Value): Primitive {
    if (value === null || typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') return value
    if (value instanceof PyTuple) return `(${value.items.map(formatValue).join(',')})`
    throw this.error(`unhashable type: '${typeName(value)}'`)
  }

  private listIndex(index: Value, length: number): number {
    const value = toInt(index, 'index')
    const resolved = value < 0 ? value + length : value
    if (resolved < 0 || resolved >= length) throw this.error(`IndexError: index ${value} is out of range for length ${length}`)
    return resolved
  }

  private sliceItems<T>(items: T[], slice: PySlice): T[] {
    const step = slice.step ?? 1
    const length = items.length
    const norm = (value: number | null, fallback: number) => value === null ? fallback : Math.max(step > 0 ? 0 : -1, Math.min(length, value < 0 ? value + length : value))
    const start = norm(slice.start, step > 0 ? 0 : length - 1)
    const stop = norm(slice.stop, step > 0 ? length : -1)
    const result: T[] = []
    for (let index = start; step > 0 ? index < stop : index > stop; index += step) result.push(items[index])
    return result
  }

  private getItem(object: Value, index: Value): Value {
    if (object instanceof PyList || object instanceof PyTuple) {
      if (index instanceof PySlice) return object instanceof PyList ? new PyList(this.sliceItems(object.items, index)) : new (object instanceof TorchSize ? TorchSize : PyTuple)(this.sliceItems(object.items, index))
      return object.items[this.listIndex(index, object.items.length)]
    }
    if (typeof object === 'string') {
      if (index instanceof PySlice) return this.sliceItems(object.split(''), index).join('')
      return object[this.listIndex(index, object.length)]
    }
    if (object instanceof PyDict) {
      const key = this.key(index)
      if (!object.map.has(key)) throw this.error(`KeyError: ${formatValue(index)}`)
      return object.map.get(key)!
    }
    if (object instanceof PyRange) return object.items[this.listIndex(index, object.items.length)]
    if (object instanceof ModuleObj && object.kind === 'user') {
      const getItem = this.bindFrom(object.cls!, '__getitem__', object)
      if (getItem !== undefined) return this.call(getItem, [index], new Map(), this.location)
    }
    if (object instanceof ModuleObj) {
      const children = [...object.children.values()]
      if (object.kind === 'moduledict') {
        const child = object.children.get(String(index))
        if (!child) throw this.error(`KeyError: '${formatValue(index)}' not in ModuleDict`)
        return child
      }
      if (index instanceof PySlice) {
        const sequence = new ModuleObj(object.kind === 'modulelist' ? 'modulelist' : 'sequential', object.className, object.site)
        this.sliceItems(children, index).forEach((child, position) => sequence.children.set(String(position), child))
        return sequence
      }
      return children[this.listIndex(index, children.length)]
    }
    if (object instanceof Tensor) {
      const items = index instanceof PyTuple && !(index instanceof TorchSize) ? index.items : [index]
      const shape = S.indexShape(object.shape, items.map((item): S.IndexItem => {
        if (item === null) return 'none'
        if (item === Ellipsis) return 'ellipsis'
        if (item instanceof PySlice) return { start: item.start, stop: item.stop, step: item.step }
        if (item instanceof Tensor) return item.dtype === 'bool' ? { array: [item.numel], consumes: Math.max(1, item.shape.length) } : { array: item.shape, consumes: 1 }
        if (item instanceof PyList || item instanceof TorchSize) {
          const nested = nestedListShape(item)
          return this.iterate(item).every((value) => typeof value === 'boolean') ? { array: [item.items.filter((value) => value === true).length], consumes: 1 } : { array: nested, consumes: 1 }
        }
        if (typeof item === 'boolean') return item ? 'none' : { array: [0], consumes: 0 }
        return toInt(item, 'index')
      }))
      const tensors = items.filter((item): item is Tensor => item instanceof Tensor)
      if (tensors.length && this.recording) return this.recordOp('index', 'generic', [object, ...tensors], new Tensor(shape, null, object.dtype), this.location.line, undefined, undefined, 0, 'index')
      return this.passThrough(object, shape, object.dtype, 'index')
    }
    if (object instanceof PyInstance) {
      if (this.atlasKind(object)) return this.datasetSample(object)
      const getItem = this.bindFrom(object.cls, '__getitem__', object)
      if (getItem !== undefined) return this.call(getItem, [index], new Map(), this.location)
    }
    if (object instanceof PyClass) return object // typing generics such as Optional[int]
    if (object === null) return null
    throw this.error(`'${typeName(object)}' object is not subscriptable`)
  }

  iterate(value: Value): Value[] {
    if (value instanceof PyList || value instanceof PyTuple) return [...value.items]
    if (value instanceof PyRange) return value.items
    if (value instanceof PyDict) return [...value.map.keys()]
    if (typeof value === 'string') return value.split('')
    if (value instanceof ModuleObj) {
      if (value.kind === 'moduledict') return [...value.children.keys()]
      if (value.kind === 'sequential' || value.kind === 'modulelist') return [...value.children.values()]
      const iter = value.kind === 'user' ? this.bindFrom(value.cls!, '__iter__', value) : undefined
      if (iter !== undefined) return this.iterate(this.call(iter, [], new Map(), this.location))
    }
    if (this.atlasKind(value) === 'DataLoader') return this.loaderBatches(value as PyInstance)
    if (value instanceof PyInstance && this.atlasKind(value) === undefined) {
      const iter = this.bindFrom(value.cls, '__iter__', value)
      if (iter !== undefined) return this.iterate(this.call(iter, [], new Map(), this.location))
      if (this.bindFrom(value.cls, '__getitem__', value) !== undefined) return [this.datasetSample(value)]
    }
    if (value instanceof Tensor) {
      if (!value.shape.length) throw this.error('Cannot iterate over a 0-d tensor')
      if (value.shape[0] > 4096) throw this.error('Iterating over a very large tensor dimension is not supported')
      return Array.from({ length: value.shape[0] }, () => new Tensor(value.shape.slice(1), this.producerOf(value), value.dtype))
    }
    throw this.error(`'${typeName(value)}' object is not iterable`)
  }

  private compare(op: string, left: Value, right: Value): boolean | Tensor {
    if (left instanceof Tensor || right instanceof Tensor) {
      if (op === 'is' || op === 'is not') return op === 'is' ? left === right : left !== right
      const a = left instanceof Tensor ? left.shape : []
      const b = right instanceof Tensor ? right.shape : []
      return new Tensor(S.broadcast(a, b, 'comparison'), this.producerOf((left instanceof Tensor ? left : right) as Tensor), 'bool')
    }
    switch (op) {
      case '==': return equals(left, right)
      case '!=': return !equals(left, right)
      case 'is': return left === right || (left === null && right === null)
      case 'is not': return !(left === right || (left === null && right === null))
      case 'in': case 'not in': {
        let found: boolean
        if (typeof right === 'string') found = right.includes(String(left))
        else if (right instanceof PyDict) found = right.map.has(this.key(left))
        else if (right instanceof ModuleObj) found = right.children.has(String(left))
        else found = this.iterate(right).some((item) => equals(item, left))
        return op === 'in' ? found : !found
      }
    }
    const a = typeof left === 'string' ? left : toNumber(left, 'comparison')
    const b = typeof right === 'string' ? right : toNumber(right, 'comparison')
    switch (op) {
      case '<': return a < b
      case '>': return a > b
      case '<=': return a <= b
      case '>=': return a >= b
    }
    return false
  }

  // ───────────────────────────── arithmetic ─────────────────────────────

  private binary(op: string, left: Value, right: Value, line: number): Value {
    if (left instanceof Tensor || right instanceof Tensor) return this.tensorBinary(op, left, right, line)
    if (typeof left === 'string' && typeof right === 'string' && op === '+') return left + right
    if (typeof left === 'string' && op === '%') { try { return percentFormat(left, right) } catch (error) { throw this.error(`ValueError: ${(error as Error).message}`) } }
    if (typeof left === 'string' && typeof right === 'number' && op === '*') return left.repeat(right)
    if ((left instanceof PyList || left instanceof PyTuple) && op === '+' && (left instanceof PyTuple ? right instanceof PyTuple : right instanceof PyList)) {
      return left instanceof PyList ? new PyList([...left.items, ...(right as PyList).items]) : new (left instanceof TorchSize && right instanceof TorchSize ? TorchSize : PyTuple)([...left.items, ...(right as PyTuple).items])
    }
    if ((left instanceof PyList || left instanceof PyTuple) && op === '*' && typeof right === 'number') {
      const items = Array.from({ length: right }, () => left.items).flat()
      return left instanceof PyList ? new PyList(items) : new PyTuple(items)
    }
    if (left instanceof PyDict && right instanceof PyDict && op === '|') return new PyDict([...left.map.entries(), ...right.map.entries()])
    const a = toNumber(left, 'left operand')
    const b = toNumber(right, 'right operand')
    switch (op) {
      case '+': return a + b
      case '-': return a - b
      case '*': return a * b
      case '/': if (b === 0) throw this.error('ZeroDivisionError: division by zero'); return a / b
      case '//': if (b === 0) throw this.error('ZeroDivisionError: integer division by zero'); return Math.floor(a / b)
      case '%': return ((a % b) + b) % b
      case '**': return a ** b
      case '<<': return a << b
      case '>>': return a >> b
      case '&': return a & b
      case '|': return a | b
      case '^': return a ^ b
    }
    throw this.error(`Unsupported operator ${op}`)
  }

  private tensorBinary(op: string, left: Value, right: Value, line: number): Value {
    const a = left instanceof Tensor ? left : null
    const b = right instanceof Tensor ? right : null
    if (op === '@') {
      if (!a || !b) throw this.error('matmul (@) needs two tensors')
      return this.recordOp('matmul', 'matmul', [a, b], new Tensor(S.matmul(a.shape, b.shape)), line)
    }
    if (!a || !b) {
      const tensor = (a ?? b)!
      return new Tensor(tensor.shape, this.producerOf(tensor), tensor.dtype)
    }
    const names: Record<string, [string, ModelNodeType]> = { '+': ['add', 'add'], '-': ['sub', 'add'], '*': ['mul', 'multiply'], '/': ['div', 'multiply'], '**': ['pow', 'multiply'], '%': ['mod', 'generic'], '&': ['and', 'generic'], '|': ['or', 'generic'] }
    const [name, type] = names[op] ?? ['op', 'generic']
    const shape = S.broadcast(a.shape, b.shape, `'${op}'`)
    const output = new Tensor(shape, null, a.dtype === 'float' || b.dtype === 'float' ? 'float' : a.dtype)
    // Treat the older operand of an addition as a residual (skip) path.
    const edgeTypes = op === '+' ? this.residualTypes([a, b], shape) : undefined
    return this.recordOp(name, type, [a, b], output, line, undefined, edgeTypes)
  }

  /** An operand that also feeds the other operand is a skip (residual) path. */
  private residualTypes(tensors: Tensor[], output: number[]): ModelEdge['type'][] | undefined {
    if (!this.recording || !this.tracer) return undefined
    const tracer = this.tracer
    const producers = tensors.map((tensor) => this.producerOf(tensor))
    if (producers.some((id) => !id || !tracer.isComputation(id)) || producers[0] === producers[1]) return undefined
    return producers.map((id, index) => {
      const other = producers[1 - index]!
      return S.fmt(tensors[index].shape) === S.fmt(output) && tracer.isAncestor(id!, other) ? 'residual' : 'data'
    })
  }

  // ───────────────────────────── training ─────────────────────────────

  private atlasKind(value: Value): string | undefined {
    if (!(value instanceof PyInstance)) return undefined
    const kind = value.attrs.get('__atlas__')
    return typeof kind === 'string' ? kind : undefined
  }

  /** Number of samples in a dataset: TensorDataset, a split, or a user Dataset with __len__. */
  private datasetSize(dataset: Value): number {
    const kind = this.atlasKind(dataset)
    if (kind === 'TensorDataset' || kind === 'Subset') return toInt(dataset instanceof PyInstance ? dataset.attrs.get('__size__') ?? 0 : 0, 'len')
    if (dataset instanceof PyInstance) {
      const len = this.bindFrom(dataset.cls, '__len__', dataset)
      if (len !== undefined) return toInt(this.call(len, [], new Map(), this.location), '__len__')
    }
    if (dataset instanceof Tensor) return dataset.shape[0] ?? 0
    return this.iterate(dataset).length
  }

  /** One sample, as the dataset's __getitem__ returns it. */
  private datasetSample(dataset: Value): Value {
    const kind = this.atlasKind(dataset)
    if (kind === 'Subset' && dataset instanceof PyInstance) return this.datasetSample(dataset.attrs.get('dataset') ?? null)
    if (kind === 'TensorDataset' && dataset instanceof PyInstance) {
      const tensors = this.iterate(dataset.attrs.get('tensors') ?? new PyTuple([]))
      return new PyTuple(tensors.map((tensor) => tensor instanceof Tensor ? new Tensor(tensor.shape.slice(1), null, tensor.dtype) : tensor))
    }
    if (dataset instanceof PyInstance) {
      const getItem = this.bindFrom(dataset.cls, '__getitem__', dataset)
      if (getItem === undefined) throw this.error(`${dataset.cls.name} needs a __getitem__ method to be used as a Dataset`)
      return this.call(getItem, [0], new Map(), this.location)
    }
    if (dataset instanceof Tensor) return new Tensor(dataset.shape.slice(1), null, dataset.dtype)
    return this.iterate(dataset)[0] ?? null
  }

  /** Stacks one sample into a batch, like torch's default collate_fn. */
  private collate(sample: Value, batch: number): Value {
    if (sample instanceof Tensor) return new Tensor([batch, ...sample.shape], null, sample.dtype)
    if (typeof sample === 'boolean') return new Tensor([batch], null, 'bool')
    if (typeof sample === 'number') return new Tensor([batch], null, Number.isInteger(sample) ? 'long' : 'float')
    if (sample instanceof PyTuple) return new PyTuple(sample.items.map((item) => this.collate(item, batch)))
    if (sample instanceof PyList) return new PyList(sample.items.map((item) => this.collate(item, batch)))
    if (sample instanceof PyDict) return new PyDict([...sample.map.entries()].map(([key, value]) => [key, this.collate(value, batch)]))
    return new PyList(Array.from({ length: batch }, () => sample))
  }

  /** The DataLoader yields one batch: one training step stands for the epoch. Its tensors enter the graph as inputs. */
  private loaderBatches(loader: PyInstance): Value[] {
    const dataset = loader.attrs.get('dataset') ?? null
    const batchSize = toInt(loader.attrs.get('batch_size') ?? 1, 'batch_size')
    const size = this.datasetSize(dataset)
    this.lastLoaderSize = Math.max(1, loader.attrs.get('drop_last') === true ? Math.floor(size / batchSize) : Math.ceil(size / batchSize))
    if (!this.rootModule) this.openTraining()
    this.muted++
    let sample: Value
    try { sample = this.datasetSample(dataset) } finally { this.muted-- }
    const batch = this.collate(sample, Math.min(batchSize, Math.max(1, size)))
    if (!this.recording || !this.tracer) return [batch]
    const tracer = this.tracer
    const tensors = tensorsIn(batch)
    const keys = batch instanceof PyDict ? [...batch.map.keys()].map(String) : []
    const names = keys.length === tensors.length ? keys : tensors.length === 2 ? ['inputs', 'targets'] : tensors.map((_, index) => tensors.length === 1 ? 'batch' : `batch_${index}`)
    const relabel = (value: Value): Value => {
      if (value instanceof Tensor) {
        const name = names[tensors.indexOf(value)]
        const node = tracer.add({ id: tracer.uniqueId(name), name, className: 'Input', type: 'tensor', inputShapes: [], outputShapes: [value.shape], params: 0, group: false, parentId: undefined, source: { ...this.location }, callSite: { ...this.location }, config: { batch_size: batchSize, samples: size } })
        this.batchNodes.add(node.id)
        return new Tensor(value.shape, node.id, value.dtype)
      }
      if (value instanceof PyTuple) return new PyTuple(value.items.map(relabel))
      if (value instanceof PyList) return new PyList(value.items.map(relabel))
      if (value instanceof PyDict) return new PyDict([...value.map.entries()].map(([key, item]) => [key, relabel(item)]))
      return value
    }
    return [relabel(batch)]
  }

  /** `for x, y in loader:` names the batch inputs x and y. */
  private nameBatch(target: Expr, scope: Scope): void {
    if (!this.batchNodes.size || !this.tracer) return
    const visit = (expr: Expr) => {
      if (expr.kind === 'tuple' || expr.kind === 'list') { expr.items.forEach(visit); return }
      if (expr.kind !== 'name') return
      const value = scope.lookup(expr.id)
      if (!(value instanceof Tensor) || !value.producer || !this.batchNodes.has(value.producer)) return
      const tracer = this.tracer!
      const node = tracer.byId.get(value.producer)
      this.batchNodes.delete(value.producer)
      if (!node) return
      // Nothing has used the batch yet, so its node can take the variable's name as its id too.
      tracer.byId.delete(node.id)
      node.id = tracer.uniqueId(expr.id)
      tracer.byId.set(node.id, node)
      node.name = expr.id
      node.variable = expr.id
      value.producer = node.id
    }
    visit(target)
  }

  /** loss.backward(): gradients flow back from the loss; recorded as a step of the training graph. */
  private backward(loss: Tensor): Value {
    if (!this.recording) return null
    const grad = this.recordOp('backward', 'optimizer', [loss], new Tensor([]), this.location.line, undefined, undefined, 0, 'loss.backward') as Tensor
    this.gradient = grad
    return null
  }

  /** optimizer.step(): the last node of the traced training step. */
  private optimizerStep(optimizer: PyInstance): Value {
    if (!this.recording) return null
    const config: TraceNode['config'] = {}
    optimizer.attrs.forEach((value, key) => {
      if (!key.startsWith('__') && key !== 'param_groups' && (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string')) config[key] = value
      if (key === 'betas' && (value instanceof PyTuple || value instanceof PyList)) config[key] = value.items.map((item) => toNumber(item, 'betas'))
    })
    const scheduler = optimizer.attrs.get('__scheduler__')
    if (typeof scheduler === 'string') config.scheduler = scheduler
    if (optimizer.attrs.get('__zero_grad__') === true) config.zero_grad = true
    const name = optimizer.cls.name
    this.recordOp('step', 'optimizer', this.gradient ? [this.gradient] : [], new Tensor([]), this.location.line, config, undefined, 0, `${name}.step`)
    const node = this.tracer!.nodes[this.tracer!.nodes.length - 1]
    node.name = `${name}.step`
    this.stepNode = node
    this.finishTrace()
    return null
  }

  // ───────────────────────────── tracing ─────────────────────────────

  private producerOf(tensor: Tensor): string | null {
    if (tensor.producer) return tensor.producer
    if (!this.recording || !this.tracer) return null
    const name = tensor.qualifiedName ?? (tensor.role === 'buffer' ? 'buffer' : tensor.role === 'parameter' ? 'parameter' : 'constant')
    const node = this.tracer.add({
      id: this.tracer.uniqueId(name), name: name.split('.').pop() ?? name, className: tensor.role === 'value' ? 'Tensor' : tensor.role === 'buffer' ? 'Buffer' : 'Parameter',
      type: tensor.role === 'value' ? 'tensor' : 'parameter', inputShapes: [], outputShapes: [tensor.shape], params: tensor.role === 'parameter' ? tensor.numel : 0,
      group: false, source: this.location.file ? { ...this.location } : undefined,
    })
    tensor.producer = node.id
    return node.id
  }

  private recordOp(op: string, type: ModelNodeType, inputs: Tensor[], output: Value, line: number, config?: TraceNode['config'], edgeTypes?: ModelEdge['type'][], params = 0, className?: string): Value {
    const outputs = tensorsIn(output)
    if (!this.recording || !this.tracer) {
      const producer = inputs.map((tensor) => tensor.producer).find(Boolean) ?? null
      outputs.forEach((tensor) => { if (!tensor.producer) tensor.producer = producer })
      return output
    }
    const tracer = this.tracer
    const producers = inputs.map((tensor) => this.producerOf(tensor))
    const site = { file: this.location.file, line, endLine: line }
    const node = tracer.add({
      id: tracer.scopedId(op), name: op, className: className ?? op, type, inputShapes: inputs.map((tensor) => tensor.shape), outputShapes: outputs.map((tensor) => tensor.shape),
      config, params, group: false, source: site, callSite: site, op,
    })
    producers.forEach((producer, index) => { if (producer) tracer.connect(producer, node.id, edgeTypes?.[index] ?? (type === 'concat' && this.isSkip(producer, producers) ? 'branch' : 'data'), inputs[index]) })
    outputs.forEach((tensor) => { tensor.producer = node.id })
    return output
  }

  private isSkip(producer: string, producers: (string | null)[]): boolean {
    const tracer = this.tracer!
    return tracer.isComputation(producer) && producers.some((other) => other && other !== producer && tracer.isAncestor(producer, other))
  }

  private passThrough(tensor: Tensor, shape: number[], dtype = tensor.dtype, op = 'reshape'): Tensor {
    const result = new Tensor(shape, this.producerOf(tensor), dtype)
    const same = shape.length === tensor.shape.length && shape.every((size, index) => size === tensor.shape[index])
    result.reshapes = same ? tensor.reshapes : [...(tensor.reshapes ?? []), op]
    return result
  }

  private callModule(module: ModuleObj, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    if (!this.rootModule && this.traceDepth === 0) return this.traceRoot(module, args, kwargs, site)
    module.callCount++
    const recording = this.recording
    const tracer = this.tracer
    const baseId = module.path || module.className.toLowerCase()
    const id = recording && tracer ? tracer.uniqueId(module.callCount > 1 ? `${baseId}@${module.callCount}` : baseId) : ''
    const inputs = [...args, ...kwargs.values()].flatMap(tensorsIn)
    const name = module.nameInParent ?? module.className

    if (module.kind === 'layer') {
      let output: Value
      try {
        output = module.layer!.forward(args, kwargs)
      } catch (error) {
        // Point shape errors at the layer definition, where its sizes are written.
        if (error instanceof PyError && error.line === undefined) { error.line = module.site.line; error.file = module.site.file }
        throw error
      }
      if (!recording || !tracer) {
        const producer = inputs.map((tensor) => tensor.producer).find(Boolean) ?? null
        tensorsIn(output).forEach((tensor) => { if (!tensor.producer) tensor.producer = producer })
        return output
      }
      const producers = inputs.map((tensor) => this.producerOf(tensor))
      const params = [...module.layer!.params.values()].reduce((sum, tensor) => sum + tensor.numel, 0)
      const node = tracer.add({
        id, name, className: module.className, type: layerType(module.className), inputShapes: inputs.map((tensor) => tensor.shape), outputShapes: tensorsIn(output).map((tensor) => tensor.shape),
        config: module.layer!.config, params, group: false, source: module.site, callSite: site,
      })
      producers.forEach((producer, index) => { if (producer) tracer.connect(producer, node.id, 'data', inputs[index]) })
      tensorsIn(output).forEach((tensor) => { tensor.producer = node.id })
      return output
    }
    if (module.kind === 'modulelist' || module.kind === 'moduledict') throw this.error(`${module.className} is a container and cannot be called; iterate over it instead`)

    let groupNode: TraceNode | undefined
    if (recording && tracer) {
      groupNode = tracer.add({
        id, name, className: module.className, type: groupType(module.className), inputShapes: inputs.map((tensor) => tensor.shape), outputShapes: [],
        params: 0, group: true, source: module.kind === 'user' && module.cls!.site.file !== PRELUDE_FILE ? module.cls!.site : module.site, callSite: site, docstring: module.cls?.docstring,
      })
      tracer.scope.push({ id: groupNode.id })
    }
    try {
      let output: Value
      if (module.kind === 'sequential') {
        output = args[0] ?? null
        for (const child of module.children.values()) output = this.callModule(child, [output], new Map(), site)
      } else {
        const forward = module.cls!.find('forward')
        if (!(forward instanceof PyFunction)) throw this.error(`${module.className} has no forward() method`)
        output = this.invoke(forward, [module, ...args], kwargs, this.ownerOf(module.cls!, 'forward'))
      }
      if (groupNode) groupNode.outputShapes = tensorsIn(output).map((tensor) => tensor.shape)
      return output
    } finally {
      if (groupNode && tracer) tracer.scope.pop()
    }
  }

  private traceRoot(module: ModuleObj, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    const training = this.training === 'open'
    this.tracer ??= new Tracer()
    this.rootModule = module
    this.traceDepth++
    const tracer = this.tracer
    // In a training step the model is one block among batch, loss and optimizer.
    const group = training && module.kind === 'user' ? tracer.add({
      id: tracer.uniqueId('model'), name: module.className, className: module.className, type: 'module', inputShapes: [...args, ...kwargs.values()].flatMap(tensorsIn).map((tensor) => tensor.shape),
      outputShapes: [], params: 0, group: true, source: module.cls!.site, callSite: site, docstring: module.cls?.docstring,
    }) : undefined
    if (group) { tracer.scope.push({ id: group.id }); this.rootGroup = group }
    try {
      const forward = module.kind === 'user' ? module.cls!.find('forward') : undefined
      const names = forward instanceof PyFunction ? forward.params.slice(1).map((param) => param.name) : ['input']
      const wrap = (value: Value, name: string): Value => {
        if (!(value instanceof Tensor)) return value
        // A batch from the DataLoader (or a tensor computed in the training step) is already in the graph.
        if (training && value.producer) return value
        const node = tracer.add({ id: tracer.uniqueId(name), name, className: 'Input', type: 'tensor', inputShapes: [], outputShapes: [value.shape], params: 0, group: false, parentId: undefined, source: site, callSite: site })
        return new Tensor(value.shape, node.id, value.dtype)
      }
      const wrappedArgs = args.map((arg, index) => wrap(arg, names[index] ?? `input_${index}`))
      const wrappedKwargs = new Map([...kwargs.entries()].map(([key, value]) => [key, wrap(value, key)]))
      const output = module.kind === 'user'
        ? this.invoke(forward as PyFunction, [module, ...wrappedArgs], wrappedKwargs, this.ownerOf(module.cls!, 'forward'))
        : this.callModuleAsChild(module, wrappedArgs, wrappedKwargs, site)
      const outputs = tensorsIn(output)
      if (group) group.outputShapes = outputs.map((tensor) => tensor.shape)
      // In a training step the outputs flow on into the loss instead of ending the graph.
      if (!training) outputs.forEach((tensor, index) => {
        const id = tracer.uniqueId(outputs.length === 1 ? 'output' : `output_${index}`)
        const node = tracer.add({ id, name: id, className: 'Output', type: 'tensor', inputShapes: [tensor.shape], outputShapes: [tensor.shape], params: 0, group: false, parentId: undefined, source: site, callSite: site })
        const producer = this.producerOf(tensor)
        if (producer) tracer.connect(producer, node.id, 'data', tensor)
      })
      return output
    } finally {
      this.traceDepth--
      if (group) tracer.scope.pop()
      if (!training) this.finishTrace()
    }
  }

  private callModuleAsChild(module: ModuleObj, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    return this.callModule(module, args, kwargs, site)
  }

  // ───────────────────────────── calls ─────────────────────────────

  call(fn: Value, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    if (++this.depth > 120) { this.depth = 0; throw this.error('Maximum call depth exceeded') }
    try {
      if (fn instanceof NativeFn) return fn.impl(args, kwargs, site)
      if (fn instanceof PyFunction) return this.invoke(fn, args, kwargs)
      if (fn instanceof BoundMethod) return this.invoke(fn.fn, fn.fn.isStatic ? args : [fn.self, ...args], kwargs, fn.owner)
      if (fn instanceof PyClass) return this.instantiate(fn, args, kwargs, site)
      if (fn instanceof ModuleObj) {
        if (this.tracer && !this.recording && this.traceDepth === 0) return this.untraced(fn, args, kwargs, site)
        return this.callModule(fn, args, kwargs, site)
      }
      if (fn === NN_MODULE) throw this.error('nn.Module cannot be instantiated directly')
      throw this.error(`'${typeName(fn)}' object is not callable`)
    } finally {
      this.depth--
    }
  }

  /** Calls after the first traced forward pass compute shapes but are not recorded. */
  private untraced(module: ModuleObj, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    this.traceDepth++
    try { return this.callModule(module, args, kwargs, site) } finally { this.traceDepth-- }
  }

  private invoke(fn: PyFunction, args: Value[], kwargs: Map<string, Value>, owner?: PyClass): Value {
    const scope = new Scope(fn.closure)
    const params = fn.params
    let position = 0
    const extraKwargs = new Map(kwargs)
    for (const param of params) {
      if (param.star === '*') { scope.vars.set(param.name, new PyTuple(args.slice(position))); position = args.length; continue }
      if (param.star === '**') { scope.vars.set(param.name, new PyDict([...extraKwargs.entries()])); extraKwargs.clear(); continue }
      if (position < args.length) { scope.vars.set(param.name, args[position++]); continue }
      if (extraKwargs.has(param.name)) { scope.vars.set(param.name, extraKwargs.get(param.name)!); extraKwargs.delete(param.name); continue }
      if (fn.defaults.has(param.name)) { scope.vars.set(param.name, fn.defaults.get(param.name)!); continue }
      throw this.error(`${fn.name}() missing required argument '${param.name}'`)
    }
    if (position < args.length) throw this.error(`${fn.name}() takes ${params.length} positional arguments but ${args.length} were given`)
    if (extraKwargs.size) throw this.error(`${fn.name}() got an unexpected keyword argument '${[...extraKwargs.keys()][0]}'`)
    if (fn.expressionBody) return this.evaluate(fn.expressionBody, scope, fn.site.file)
    this.frames.push({ owner, self: args[0] })
    try {
      this.execBlock(fn.body ?? [], scope, fn.site.file)
      return null
    } catch (signal) {
      if (signal instanceof ReturnSignal) return signal.value
      throw signal
    } finally {
      this.frames.pop()
    }
  }

  private instantiate(cls: PyClass, args: Value[], kwargs: Map<string, Value>, site: Site): Value {
    if (cls.isModule) {
      const module = new ModuleObj('user', cls.name, site, cls)
      const init = cls.find('__init__')
      if (init instanceof PyFunction) this.invoke(init, [module, ...args], kwargs, this.ownerOf(cls, '__init__'))
      return module
    }
    const instance = new PyInstance(cls)
    if (cls.isDataclass || cls.bases.some((base) => base instanceof PyClass && base.isDataclass)) {
      const fields = this.allFields(cls)
      const remaining = new Map(kwargs)
      fields.forEach((field, index) => {
        let value: Value | undefined = index < args.length ? args[index] : remaining.get(field.name)
        remaining.delete(field.name)
        if (value === undefined && field.factory) value = this.call(field.factory, [], new Map(), site)
        if (value === undefined) value = field.defaultValue
        if (value === undefined) throw this.error(`${cls.name}() missing required argument '${field.name}'`)
        instance.attrs.set(field.name, value)
      })
      if (remaining.size) throw this.error(`${cls.name}() got an unexpected keyword argument '${[...remaining.keys()][0]}'`)
      const post = cls.find('__post_init__')
      if (post instanceof PyFunction) this.invoke(post, [instance], new Map(), cls)
      return instance
    }
    const init = cls.find('__init__')
    if (init instanceof PyFunction) this.invoke(init, [instance, ...args], kwargs, this.ownerOf(cls, '__init__'))
    else if (args.length && cls.bases.some((base) => base instanceof NativeFn && base.name === 'Exception')) instance.attrs.set('message', args[0])
    return instance
  }

  private allFields(cls: PyClass): PyClass['fields'] {
    const inherited = cls.bases.flatMap((base) => base instanceof PyClass ? this.allFields(base) : [])
    return [...inherited.filter((field) => !cls.fields.some((own) => own.name === field.name)), ...cls.fields]
  }

  private ownerOf(cls: PyClass, name: string): PyClass {
    if (cls.attrs.has(name)) return cls
    for (const base of cls.bases) if (base instanceof PyClass && base.find(name) !== undefined) return this.ownerOf(base, name)
    return cls
  }

  // ───────────────────────────── attributes ─────────────────────────────

  private bindFrom(cls: PyClass, name: string, self: Value): Value | undefined {
    const value = cls.find(name)
    if (value === undefined) return undefined
    if (value instanceof PyFunction) {
      const owner = this.ownerOf(cls, name)
      if (value.isProperty) return this.invoke(value, [self], new Map(), owner)
      if (value.isStatic) return value
      if (value.isClassMethod) return new BoundMethod(cls as unknown as Value, value, owner)
      return new BoundMethod(self, value, owner)
    }
    return value
  }

  getAttr(object: Value, name: string): Value {
    if (object instanceof ModuleObj) return this.moduleAttr(object, name)
    if (object instanceof Tensor) return this.tensorAttr(object, name)
    if (object instanceof PyInstance) {
      if (object.attrs.has(name)) return object.attrs.get(name)!
      const value = this.bindFrom(object.cls, name, object)
      if (value !== undefined) return value
      throw this.error(`AttributeError: '${object.cls.name}' object has no attribute '${name}'`)
    }
    if (object instanceof PyClass) {
      const value = object.find(name)
      if (value === undefined) throw this.error(`AttributeError: type '${object.name}' has no attribute '${name}'`)
      if (value instanceof PyFunction && value.isClassMethod) return new BoundMethod(object as unknown as Value, value, object)
      return value
    }
    if (object instanceof Namespace) {
      if (object.attrs.has(name)) return object.attrs.get(name)!
      if (object.permissive) return noop(`${object.name}.${name}`)
      throw this.error(`${object.name}.${name} is not supported by the Atlas tracer`)
    }
    if (object instanceof SuperProxy) {
      for (const base of object.cls.bases) {
        if (base instanceof PyClass) {
          const value = this.bindFrom(base, name, object.self)
          if (value !== undefined) return value
        }
        if (base === NN_MODULE && name === '__init__') return noop('Module.__init__')
      }
      if (name === '__init__') return noop('object.__init__')
      throw this.error(`super() has no attribute '${name}'`)
    }
    if (object instanceof Dtype) {
      if (name === 'is_floating_point') return object.kind === 'float'
      return object.name
    }
    return this.valueMethod(object, name)
  }

  private moduleAttr(module: ModuleObj, name: string): Value {
    if (module.children.has(name)) return module.children.get(name)!
    if (module.attrs.has(name)) return module.attrs.get(name)!
    if (module.layer?.params.has(name)) return module.layer.params.get(name)!
    if (module.layer?.props.has(name)) return module.layer.props.get(name)!
    if (module.cls) {
      const value = this.bindFrom(module.cls, name, module)
      if (value !== undefined) return value
    }
    const native = (impl: NativeImpl) => new NativeFn(`${module.className}.${name}`, impl)
    switch (name) {
      case 'parameters': return native(() => new PyList(module.parameters()))
      case 'named_parameters': return native(() => new PyList(module.parameters().map((tensor) => tupleOf([tensor.qualifiedName ?? 'param', tensor]))))
      case 'children': return native(() => new PyList([...module.children.values()]))
      case 'named_children': return native(() => new PyList([...module.children.entries()].map(([key, child]) => tupleOf([key, child]))))
      case 'modules': return native(() => {
        const all: ModuleObj[] = []
        const visit = (item: ModuleObj) => { all.push(item); item.children.forEach(visit) }
        visit(module)
        return new PyList(all)
      })
      case 'apply': return native(() => module)
      case 'register_buffer': return native((args) => {
        const key = String(args[0])
        const value = args[1]
        if (value instanceof Tensor) { value.role = 'buffer'; value.label = key; value.owner = module }
        module.attrs.set(key, value ?? null)
        return null
      })
      case 'register_parameter': return native((args) => { this.setAttr(module, String(args[0]), args[1] ?? null); return null })
      case 'add_module': case 'register_module': return native((args) => { if (args[1] instanceof ModuleObj) module.addChild(String(args[0]), args[1]); return null })
      case 'to': case 'cuda': case 'cpu': case 'eval': case 'train': case 'float': case 'half': case 'requires_grad_': case 'double': return native(() => module)
      case 'zero_grad': return native(() => null)
      case 'training': return false
      case 'forward': return native((args, kwargs, site) => this.call(module, args, kwargs, site))
      case 'state_dict': return native(() => new PyDict())
      case 'load_state_dict': return native(() => null)
      case 'append': return native((args) => { if (args[0] instanceof ModuleObj) module.addChild(String(module.children.size), args[0]); return module })
      case 'extend': return native((args) => { this.iterate(args[0]).forEach((item) => { if (item instanceof ModuleObj) module.addChild(String(module.children.size), item) }); return module })
      case 'insert': return native((args) => {
        const entries = [...module.children.values()]
        if (args[1] instanceof ModuleObj) entries.splice(toInt(args[0], 'index'), 0, args[1])
        module.children.clear()
        entries.forEach((item, index) => { module.children.set(String(index), item); if (item.parent === module) item.nameInParent = String(index) })
        return null
      })
      case 'keys': return native(() => new PyList([...module.children.keys()]))
      case 'values': return native(() => new PyList([...module.children.values()]))
      case 'items': return native(() => new PyList([...module.children.entries()].map(([key, child]) => tupleOf([key, child]))))
    }
    throw this.error(`AttributeError: '${module.className}' object has no attribute '${name}'`)
  }

  private setAttr(object: Value, name: string, value: Value): void {
    if (object instanceof ModuleObj) {
      if (value instanceof ModuleObj) { object.attrs.delete(name); object.addChild(name, value); return }
      if (object.layer?.params.has(name) && value instanceof Tensor) { object.layer.params.set(name, value); return }
      if (value instanceof Tensor && value.role === 'parameter' && !value.owner) { value.label = name; value.owner = object }
      object.children.delete(name)
      object.attrs.set(name, value)
      return
    }
    if (object instanceof PyInstance) { object.attrs.set(name, value); return }
    if (object instanceof PyClass) { object.attrs.set(name, value); return }
    if (object instanceof Tensor) return
    throw this.error(`Cannot set attribute '${name}' on ${typeName(object)}`)
  }

  private valueMethod(object: Value, name: string): Value {
    const native = (impl: NativeImpl) => new NativeFn(name, impl)
    if (object instanceof PyList) {
      switch (name) {
        case 'append': return native((args) => { object.items.push(args[0]); return null })
        case 'extend': return native((args) => { object.items.push(...this.iterate(args[0])); return null })
        case 'pop': return native((args) => object.items.splice(args.length ? this.listIndex(args[0], object.items.length) : object.items.length - 1, 1)[0] ?? null)
        case 'insert': return native((args) => { object.items.splice(toInt(args[0], 'index'), 0, args[1]); return null })
        case 'reverse': return native(() => { object.items.reverse(); return null })
        case 'copy': return native(() => new PyList([...object.items]))
        case 'index': return native((args) => object.items.findIndex((item) => equals(item, args[0])))
        case 'count': return native((args) => object.items.filter((item) => equals(item, args[0])).length)
      }
    }
    if (object instanceof PyTuple) {
      if (name === 'index') return native((args) => object.items.findIndex((item) => equals(item, args[0])))
      if (name === 'count') return native((args) => object.items.filter((item) => equals(item, args[0])).length)
      if (name === 'numel') return native(() => object.items.reduce((total: number, item) => total * toNumber(item, 'size'), 1))
    }
    if (object instanceof PyDict) {
      switch (name) {
        case 'get': return native((args) => object.map.has(this.key(args[0])) ? object.map.get(this.key(args[0]))! : args[1] ?? null)
        case 'items': return native(() => new PyList([...object.map.entries()].map(([key, value]) => tupleOf([key, value]))))
        case 'keys': return native(() => new PyList([...object.map.keys()]))
        case 'values': return native(() => new PyList([...object.map.values()]))
        case 'update': return native((args, kwargs) => { if (args[0] instanceof PyDict) args[0].map.forEach((value, key) => object.map.set(key, value)); kwargs.forEach((value, key) => object.map.set(key, value)); return null })
        case 'pop': return native((args) => { const key = this.key(args[0]); const value = object.map.get(key) ?? args[1] ?? null; object.map.delete(key); return value })
        case 'setdefault': return native((args) => { const key = this.key(args[0]); if (!object.map.has(key)) object.map.set(key, args[1] ?? null); return object.map.get(key)! })
        case 'copy': return native(() => new PyDict([...object.map.entries()]))
      }
    }
    if (typeof object === 'string') {
      switch (name) {
        case 'format': return native((args, kwargs) => { try { return strFormat(object, args, kwargs, (value, path) => this.formatLookup(value, path)) } catch (error) { throw this.error(`ValueError: ${(error as Error).message}`) } })
        case 'title': return native(() => object.replace(/\b\w/g, (char) => char.toUpperCase()))
        case 'capitalize': return native(() => object.charAt(0).toUpperCase() + object.slice(1).toLowerCase())
        case 'lstrip': return native((args) => args[0] ? object.replace(new RegExp(`^[${String(args[0]).replace(/[\]\\^-]/g, '\\$&')}]+`), '') : object.trimStart())
        case 'rstrip': return native((args) => args[0] ? object.replace(new RegExp(`[${String(args[0]).replace(/[\]\\^-]/g, '\\$&')}]+$`), '') : object.trimEnd())
        case 'ljust': return native((args) => object.padEnd(toInt(args[0], 'width'), String(args[1] ?? ' ')))
        case 'rjust': return native((args) => object.padStart(toInt(args[0], 'width'), String(args[1] ?? ' ')))
        case 'center': return native((args) => formatSpec(object, `${String(args[1] ?? ' ')}^${toInt(args[0], 'width')}`))
        case 'zfill': return native((args) => formatSpec(object, `0>${toInt(args[0], 'width')}`))
        case 'find': return native((args) => object.indexOf(String(args[0])))
        case 'count': return native((args) => object.split(String(args[0])).length - 1)
        case 'isdigit': return native(() => /^\d+$/.test(object))
        case 'splitlines': return native(() => new PyList(object.split('\n')))
        case 'join': return native((args) => this.iterate(args[0]).map((item) => String(item)).join(object))
        case 'upper': return native(() => object.toUpperCase())
        case 'lower': return native(() => object.toLowerCase())
        case 'startswith': return native((args) => object.startsWith(String(args[0])))
        case 'endswith': return native((args) => object.endsWith(String(args[0])))
        case 'split': return native((args) => new PyList(object.split(typeof args[0] === 'string' ? args[0] : /\s+/).filter((part) => part !== '' || typeof args[0] === 'string')))
        case 'replace': return native((args) => object.split(String(args[0])).join(String(args[1])))
        case 'strip': return native(() => object.trim())
      }
    }
    if (typeof object === 'number' && name === 'is_integer') return native(() => Number.isInteger(object))
    throw this.error(`AttributeError: '${typeName(object)}' object has no attribute '${name}'`)
  }

  // ───────────────────────────── tensors ─────────────────────────────

  private dims(args: Value[], kwargs: Map<string, Value>, key = 'dim'): Value { return args[0] ?? kwargs.get(key) ?? null }

  private tensorAttr(tensor: Tensor, name: string): Value {
    const native = (impl: NativeImpl) => new NativeFn(`Tensor.${name}`, impl)
    const pass = (shape: number[], dtype = tensor.dtype) => this.passThrough(tensor, shape, dtype, name)
    switch (name) {
      case 'shape': return new TorchSize(tensor.shape)
      case 'ndim': return tensor.shape.length
      case 'dtype': return new Dtype(tensor.dtype === 'float' ? 'torch.float32' : tensor.dtype === 'long' ? 'torch.int64' : 'torch.bool', tensor.dtype)
      case 'device': return 'cpu'
      case 'requires_grad': return tensor.role === 'parameter'
      case 'T': case 'mT': return pass([...tensor.shape].reverse())
      case 'data': return tensor
      case 'size': return native((args) => args.length ? tensor.shape[S.normalizeDim(toInt(args[0], 'dim'), tensor.shape.length)] : new TorchSize(tensor.shape))
      case 'dim': return native(() => tensor.shape.length)
      case 'numel': case 'nelement': return native(() => tensor.numel)
      case 'item': return native(() => 0)
      case 'backward': return native(() => this.backward(tensor))
      case 'retain_grad': case 'zero_': return native(() => null)
      case 'grad': return null
      case 'tolist': return native(() => new PyList([]))
      case 'view': case 'reshape': return native((args) => {
        if (args[0] instanceof Dtype) return tensor
        return pass(S.reshape(tensor.shape, shapeArgs(args, name)))
      })
      case 'view_as': case 'reshape_as': case 'expand_as': case 'type_as': return native((args) => pass((args[0] as Tensor).shape))
      case 'flatten': return native((args, kwargs) => pass(S.flatten(tensor.shape, toInt(args[0] ?? kwargs.get('start_dim') ?? 0, 'start_dim'), toInt(args[1] ?? kwargs.get('end_dim') ?? -1, 'end_dim'))))
      case 'unflatten': return native((args) => {
        const dim = S.normalizeDim(toInt(args[0], 'dim'), tensor.shape.length)
        const sizes = toInts(args[1], 'sizes')
        return pass([...tensor.shape.slice(0, dim), ...S.reshape([tensor.shape[dim]], sizes), ...tensor.shape.slice(dim + 1)])
      })
      case 'transpose': case 'swapaxes': return native((args) => pass(S.transpose(tensor.shape, toInt(args[0], 'dim0'), toInt(args[1], 'dim1'))))
      case 't': return native(() => pass([...tensor.shape].reverse()))
      case 'permute': return native((args) => pass(S.permute(tensor.shape, shapeArgs(args, 'dims'))))
      case 'contiguous': case 'clone': case 'detach': case 'cpu': case 'cuda': case 'requires_grad_': case 'float': case 'double': case 'half': case 'bfloat16':
        return native(() => pass(tensor.shape, name === 'float' || name === 'double' || name === 'half' || name === 'bfloat16' ? 'float' : tensor.dtype))
      case 'long': case 'int': return native(() => pass(tensor.shape, 'long'))
      case 'bool': return native(() => pass(tensor.shape, 'bool'))
      case 'to': case 'type': return native((args, kwargs) => {
        const dtype = [...args, ...kwargs.values()].find((item) => item instanceof Dtype) as Dtype | undefined
        return pass(tensor.shape, dtype?.kind ?? tensor.dtype)
      })
      case 'unsqueeze': return native((args) => {
        const dim = toInt(args[0], 'dim')
        const axis = dim < 0 ? dim + tensor.shape.length + 1 : dim
        const shape = [...tensor.shape]
        shape.splice(axis, 0, 1)
        return pass(shape)
      })
      case 'squeeze': return native((args) => {
        if (!args.length) return pass(tensor.shape.filter((size) => size !== 1))
        const axis = S.normalizeDim(toInt(args[0], 'dim'), tensor.shape.length)
        return pass(tensor.shape.filter((size, index) => index !== axis || size !== 1))
      })
      case 'expand': return native((args) => {
        const sizes = shapeArgs(args, 'sizes')
        const offset = sizes.length - tensor.shape.length
        return pass(sizes.map((size, index) => size === -1 ? tensor.shape[index - offset] : size))
      })
      case 'repeat': return native((args) => {
        const sizes = shapeArgs(args, 'repeats')
        const padded = [...Array(sizes.length - tensor.shape.length).fill(1), ...tensor.shape]
        return pass(padded.map((size, index) => size * sizes[index]))
      })
      case 'chunk': return native((args, kwargs) => this.chunk(tensor, toInt(args[0] ?? kwargs.get('chunks')!, 'chunks'), toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim')))
      case 'split': return native((args, kwargs) => this.split(tensor, args[0], toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim')))
      case 'unbind': return native((args) => {
        const axis = S.normalizeDim(toInt(args[0] ?? 0, 'dim'), tensor.shape.length)
        return new PyTuple(Array.from({ length: tensor.shape[axis] }, () => pass(tensor.shape.filter((_, index) => index !== axis))))
      })
      case 'mean': case 'sum': case 'amax': case 'amin': case 'std': case 'var': case 'prod': case 'logsumexp': case 'norm':
        return native((args, kwargs) => this.reduction(name, tensor, this.dims(name === 'norm' ? args.slice(1) : args, kwargs), Boolean(kwargs.get('keepdim') ?? args[name === 'norm' ? 2 : 1] ?? false)))
      case 'max': case 'min': return native((args, kwargs) => {
        const dim = this.dims(args, kwargs)
        if (dim === null || dim instanceof Tensor) return dim instanceof Tensor ? this.tensorFunction('maximum', [tensor, dim], new Map()) : this.reduction(name, tensor, null, false)
        const values = this.reduction(name, tensor, dim, Boolean(kwargs.get('keepdim') ?? args[1] ?? false))
        return new PyTuple([values, new Tensor((values as Tensor).shape, (values as Tensor).producer, 'long')])
      })
      case 'argmax': case 'argmin': return native((args, kwargs) => {
        const dim = this.dims(args, kwargs)
        return pass(S.reduce(tensor.shape, dim === null ? null : toInt(dim, 'dim'), Boolean(kwargs.get('keepdim') ?? false)), 'long')
      })
      case 'softmax': case 'log_softmax': return native(() => this.recordOp(name, 'activation', [tensor], new Tensor(tensor.shape), this.location.line))
      case 'exp': case 'log': case 'sigmoid': case 'tanh': case 'relu': case 'sqrt': case 'rsqrt': case 'sin': case 'cos': case 'square': case 'neg': case 'abs': case 'reciprocal':
        return native(() => ['neg', 'abs', 'square'].includes(name) ? pass(tensor.shape) : this.recordOp(name, 'activation', [tensor], new Tensor(tensor.shape), this.location.line))
      case 'masked_fill': return native((args) => {
        const mask = args[0]
        if (!(mask instanceof Tensor)) return pass(tensor.shape)
        S.broadcast(tensor.shape, mask.shape, 'masked_fill')
        return this.recordOp('mask', 'generic', [tensor, mask], new Tensor(tensor.shape, null, tensor.dtype), this.location.line, undefined, ['data', 'condition'], 0, 'masked_fill')
      })
      case 'pow': case 'clamp': case 'clip': case 'fill_': case 'normal_': case 'uniform_': case 'add_': case 'mul_': case 'copy_': case 'flip': case 'roll': case 'cumsum': case 'round': case 'floor': case 'ceil':
        return native(() => pass(tensor.shape))
      case 'tril': case 'triu': return native(() => pass(tensor.shape))
      case 'matmul': case 'mm': case 'bmm': return native((args) => this.tensorBinary('@', tensor, args[0], this.location.line))
      case 'add': case 'sub': case 'mul': case 'div': case 'multiply': case 'subtract': case 'divide':
        return native((args) => this.tensorBinary({ add: '+', sub: '-', mul: '*', div: '/', multiply: '*', subtract: '-', divide: '/' }[name]!, tensor, args[0], this.location.line))
      case 'new_zeros': case 'new_ones': case 'new_empty': case 'new_full': return native((args) => new Tensor(shapeArgs(args.slice(0, name === 'new_full' ? 1 : args.length), 'size'), null, tensor.dtype))
      case 'masked_fill_': return native(() => tensor)
      case 'is_floating_point': return native(() => tensor.dtype === 'float')
      case 'topk': case 'sort': case 'kthvalue': case 'median': case 'mode': case 'cummax': case 'cummin': case 'aminmax':
        return native((args, kwargs) => this.selectWithIndices(name, tensor, args, kwargs))
      case 'argsort': return native((args, kwargs) => {
        const dim = toInt(args[0] ?? kwargs.get('dim') ?? -1, 'dim')
        return this.recordOp('argsort', 'generic', [tensor], new Tensor(tensor.shape, null, 'long'), this.location.line, { dim })
      })
      case 'gather': return native((args, kwargs) => {
        const index = this.asTensor(args[1] ?? kwargs.get('index'), 'gather index')
        if (index.shape.length !== tensor.shape.length) throw this.error(`gather: index has ${index.shape.length} dimensions but the input has ${tensor.shape.length}`)
        return this.recordOp('gather', 'generic', [tensor, index], new Tensor(index.shape, null, tensor.dtype), this.location.line, { dim: toInt(args[0] ?? kwargs.get('dim')!, 'dim') })
      })
      case 'scatter': case 'scatter_': case 'scatter_add': case 'scatter_add_': case 'scatter_reduce': case 'scatter_reduce_':
      case 'index_add': case 'index_add_': case 'index_copy': case 'index_copy_': case 'index_fill': case 'index_fill_': case 'index_put': case 'index_put_': case 'masked_scatter': case 'masked_scatter_':
        return native((args) => {
          const others = args.filter((item): item is Tensor => item instanceof Tensor)
          const output = this.recordOp(name.replace(/_$/, ''), 'generic', [tensor, ...others], new Tensor(tensor.shape, null, tensor.dtype), this.location.line)
          if (name.endsWith('_') && output instanceof Tensor) { tensor.producer = output.producer; return tensor }
          return output
        })
      case 'index_select': return native((args, kwargs) => {
        const axis = S.normalizeDim(toInt(args[0] ?? kwargs.get('dim')!, 'dim'), tensor.shape.length)
        const index = this.asTensor(args[1] ?? kwargs.get('index'), 'index_select index')
        const shape = [...tensor.shape]
        shape[axis] = index.numel
        return this.recordOp('index_select', 'generic', [tensor, index], new Tensor(shape, null, tensor.dtype), this.location.line, { dim: axis })
      })
      case 'masked_select': return native((args) => {
        const mask = this.asTensor(args[0], 'masked_select mask')
        return this.recordOp('masked_select', 'generic', [tensor, mask], new Tensor([S.broadcast(tensor.shape, mask.shape, 'masked_select').reduce((p, v) => p * v, 1)], null, tensor.dtype), this.location.line)
      })
      case 'nonzero': case 'argwhere': return native((_, kwargs) => {
        if (isTruthy(kwargs.get('as_tuple') ?? false)) return new PyTuple(tensor.shape.map(() => new Tensor([tensor.numel], null, 'long')))
        return this.recordOp('nonzero', 'generic', [tensor], new Tensor([tensor.numel, tensor.shape.length], null, 'long'), this.location.line)
      })
      case 'any': case 'all': return native((args, kwargs) => {
        const dim = this.dims(args, kwargs)
        return pass(S.reduce(tensor.shape, dim === null ? null : toInt(dim, 'dim'), Boolean(kwargs.get('keepdim') ?? false)), 'bool')
      })
      case 'eq': case 'ne': case 'lt': case 'le': case 'gt': case 'ge': case 'greater': case 'less': case 'equal': case 'logical_and': case 'logical_or': case 'logical_xor': case 'isclose':
        return native((args) => {
          if (name === 'equal') return true
          const other = args[0]
          const shape = other instanceof Tensor ? S.broadcast(tensor.shape, other.shape, name) : tensor.shape
          return this.recordOp(name, 'generic', other instanceof Tensor ? [tensor, other] : [tensor], new Tensor(shape, null, 'bool'), this.location.line)
        })
      case 'logical_not': case 'isnan': case 'isinf': case 'isfinite': case 'signbit': return native(() => pass(tensor.shape, 'bool'))
      case 'sign': case 'sgn': case 'trunc': case 'frac': case 'log1p': case 'expm1': case 'erf': case 'erfinv': case 'log2': case 'log10': case 'nan_to_num': case 'nan_to_num_':
      case 'clamp_': case 'clamp_min': case 'clamp_max': case 'clamp_min_': case 'clamp_max_': case 'abs_': case 'neg_': case 'sub_': case 'div_': case 'pow_': case 'sqrt_': case 'exp_': case 'log_': case 'sigmoid_': case 'tanh_': case 'relu_': case 'lerp': case 'lerp_':
      case 'fill_diagonal_': case 'bernoulli': case 'bernoulli_': case 'random_': case 'exponential_': case 'geometric_': case 'cauchy_': case 'log_normal_': case 'detach_': case 'cumprod': case 'logcumsumexp':
      case 'addcmul': case 'addcmul_': case 'addcdiv': case 'addcdiv_': case 'clone_': case 'floor_': case 'ceil_': case 'round_': case 'tril_': case 'triu_': case 'square_': case 'reciprocal_': case 'sin_': case 'cos_': case 'fmod': case 'remainder':
        return native(() => pass(tensor.shape))
      case 'squeeze_': case 'unsqueeze_': case 't_': case 'transpose_': case 'resize_': case 'resize_as_':
        return native((args, kwargs, site) => {
          const reshaped = name === 'resize_' ? shapeArgs(args, 'size') : name === 'resize_as_' ? this.asTensor(args[0], 'resize_as_').shape : (this.call(this.tensorAttr(tensor, name.slice(0, -1)), args, kwargs, site) as Tensor).shape
          tensor.shape = reshaped
          return tensor
        })
      case 'narrow': return native((args) => {
        const axis = S.normalizeDim(toInt(args[0], 'dim'), tensor.shape.length)
        const shape = [...tensor.shape]
        shape[axis] = toInt(args[2], 'length')
        return pass(shape)
      })
      case 'select': return native((args) => {
        const axis = S.normalizeDim(toInt(args[0], 'dim'), tensor.shape.length)
        return pass(tensor.shape.filter((_, index) => index !== axis))
      })
      case 'repeat_interleave': return native((args, kwargs) => pass(this.repeatInterleave(tensor, args[0] ?? kwargs.get('repeats') ?? 1, args[1] ?? kwargs.get('dim') ?? null)))
      case 'tile': return native((args) => {
        const reps = shapeArgs(args, 'dims')
        const rank = Math.max(reps.length, tensor.shape.length)
        const padded = [...Array(rank - tensor.shape.length).fill(1), ...tensor.shape]
        const factors = [...Array(rank - reps.length).fill(1), ...reps]
        return pass(padded.map((size, index) => size * factors[index]))
      })
      case 'diagonal': return native((args, kwargs) => pass(this.diagonalShape(tensor.shape, toInt(args[0] ?? kwargs.get('offset') ?? 0, 'offset'), toInt(args[1] ?? kwargs.get('dim1') ?? 0, 'dim1'), toInt(args[2] ?? kwargs.get('dim2') ?? 1, 'dim2'))))
      case 'diag': return native((args) => pass(tensor.shape.length === 1 ? [tensor.shape[0] + Math.abs(toInt(args[0] ?? 0, 'diagonal')), tensor.shape[0] + Math.abs(toInt(args[0] ?? 0, 'diagonal'))] : this.diagonalShape(tensor.shape, toInt(args[0] ?? 0, 'diagonal'), 0, 1)))
      case 'diff': return native((args, kwargs) => {
        const n = toInt(args[0] ?? kwargs.get('n') ?? 1, 'n')
        const axis = S.normalizeDim(toInt(args[1] ?? kwargs.get('dim') ?? -1, 'dim'), tensor.shape.length)
        const shape = [...tensor.shape]
        shape[axis] = Math.max(0, shape[axis] - n)
        return pass(shape)
      })
      case 'unfold': return native((args) => {
        const axis = S.normalizeDim(toInt(args[0], 'dimension'), tensor.shape.length)
        const size = toInt(args[1], 'size')
        const step = toInt(args[2], 'step')
        const shape = [...tensor.shape]
        shape[axis] = Math.floor((shape[axis] - size) / step) + 1
        return pass([...shape, size])
      })
      case 'outer': case 'ger': return native((args) => this.outer(tensor, this.asTensor(args[0], 'outer')))
      case 'dot': case 'vdot': case 'inner': return native((args) => this.recordOp(name, 'matmul', [tensor, this.asTensor(args[0], name)], new Tensor([]), this.location.line))
      case 'cross': return native((args) => this.recordOp('cross', 'multiply', [tensor, this.asTensor(args[0], 'cross')], new Tensor(tensor.shape), this.location.line))
      case 'multinomial': return native((args, kwargs) => pass([...tensor.shape.slice(0, -1), toInt(args[0] ?? kwargs.get('num_samples')!, 'num_samples')], 'long'))
      case 'ravel': return native(() => pass([tensor.numel]))
      case 'movedim': case 'moveaxis': return native((args) => pass(this.movedim(tensor.shape, args[0], args[1])))
      case 'broadcast_to': return native((args) => pass(shapeArgs(args, 'size').map((size, index, sizes) => size === -1 ? tensor.shape[index - (sizes.length - tensor.shape.length)] : size)))
      case 'new_tensor': return native((args) => new Tensor(nestedListShape(args[0]), null, tensor.dtype))
      case 'numpy': case 'cpu_': case 'pin_memory': case 'share_memory_': return native(() => tensor)
      case 'is_contiguous': case 'is_cuda': return native(() => name === 'is_contiguous')
      case 'element_size': return native(() => tensor.dtype === 'long' ? 8 : tensor.dtype === 'bool' ? 1 : 4)
      case 'stride': return native((args) => {
        const strides = tensor.shape.map((_, index) => tensor.shape.slice(index + 1).reduce((p, v) => p * v, 1))
        return args.length ? strides[S.normalizeDim(toInt(args[0], 'dim'), strides.length)] : new PyTuple(strides)
      })
      case 'split_with_sizes': return native((args, kwargs) => this.split(tensor, args[0], toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim')))
      case 'std_mean': case 'var_mean': return native((args, kwargs) => {
        const result = this.reduction(name.split('_')[0], tensor, this.dims(args, kwargs), Boolean(kwargs.get('keepdim') ?? false)) as Tensor
        return new PyTuple([result, new Tensor(result.shape, result.producer)])
      })
      case 'softmax_': return native(() => tensor)
      case 'view_as_complex': case 'view_as_real': return native(() => pass(name === 'view_as_complex' ? tensor.shape.slice(0, -1) : [...tensor.shape, 2]))
      case 'real': case 'imag': return pass(tensor.shape)
    }
    throw this.error(`AttributeError: 'Tensor' object has no attribute '${name}' (not supported by the Atlas tracer)`)
  }

  private asTensor(value: Value | undefined, what: string): Tensor {
    if (!(value instanceof Tensor)) throw this.error(`${what} expects a Tensor, got ${value === undefined ? 'nothing' : formatValue(value)}`)
    return value
  }

  /** topk, sort, kthvalue, median, mode, cummax …: (values, indices) pairs. */
  private selectWithIndices(name: string, tensor: Tensor, args: Value[], kwargs: Map<string, Value>): Value {
    const line = this.location.line
    const rank = tensor.shape.length
    let shape: number[]
    let config: TraceNode['config'] = {}
    if (name === 'topk') {
      const k = toInt(args[0] ?? kwargs.get('k')!, 'k')
      const axis = S.normalizeDim(toInt(args[1] ?? kwargs.get('dim') ?? -1, 'dim'), rank)
      if (k > tensor.shape[axis]) throw this.error(`topk: k=${k} is larger than dimension ${axis} of size ${tensor.shape[axis]}`)
      shape = [...tensor.shape]
      shape[axis] = k
      config = { k, dim: axis }
    } else if (name === 'sort' || name === 'cummax' || name === 'cummin') {
      shape = [...tensor.shape]
      config = { dim: toInt(args[0] ?? kwargs.get('dim') ?? -1, 'dim') }
    } else if (name === 'aminmax') {
      const dim = kwargs.get('dim') ?? null
      const reduced = S.reduce(tensor.shape, dim === null ? null : toInt(dim, 'dim'), Boolean(kwargs.get('keepdim') ?? false))
      const values = this.recordOp('aminmax', 'pool', [tensor], new Tensor(reduced, null, tensor.dtype), line) as Tensor
      return new PyTuple([values, new Tensor(reduced, values.producer, tensor.dtype)])
    } else {
      const dimArg = name === 'kthvalue' ? args[1] ?? kwargs.get('dim') ?? -1 : args[0] ?? kwargs.get('dim') ?? null
      if (dimArg === null) return this.reduction(name, tensor, null, false)
      shape = S.reduce(tensor.shape, toInt(dimArg, 'dim'), Boolean(kwargs.get('keepdim') ?? args[name === 'kthvalue' ? 2 : 1] ?? false))
      config = { dim: toInt(dimArg, 'dim') }
    }
    const values = this.recordOp(name, name === 'sort' || name === 'topk' ? 'generic' : 'pool', [tensor], new Tensor(shape, null, tensor.dtype), line, config) as Tensor
    return new PyTuple([values, new Tensor(shape, values.producer, 'long')])
  }

  private repeatInterleave(tensor: Tensor, repeats: Value, dim: Value): number[] {
    // A tensor of per-element repeats depends on its values; the size is then left unchanged.
    const total = (size: number) => repeats instanceof Tensor ? size : size * toInt(repeats, 'repeats')
    if (dim === null) return [total(tensor.numel)]
    const axis = S.normalizeDim(toInt(dim, 'dim'), tensor.shape.length)
    const shape = [...tensor.shape]
    shape[axis] = total(shape[axis])
    return shape
  }

  private diagonalShape(shape: number[], offset: number, dim1: number, dim2: number): number[] {
    const a = S.normalizeDim(dim1, shape.length)
    const b = S.normalizeDim(dim2, shape.length)
    const length = Math.max(0, offset >= 0 ? Math.min(shape[a], shape[b] - offset) : Math.min(shape[a] + offset, shape[b]))
    return [...shape.filter((_, index) => index !== a && index !== b), length]
  }

  private outer(a: Tensor, b: Tensor): Tensor {
    if (a.shape.length !== 1 || b.shape.length !== 1) throw this.error(`outer expects two 1-D tensors, got ${S.fmt(a.shape)} and ${S.fmt(b.shape)}`)
    return this.recordOp('outer', 'multiply', [a, b], new Tensor([a.shape[0], b.shape[0]]), this.location.line) as Tensor
  }

  private movedim(shape: number[], source: Value, destination: Value): number[] {
    const from = toInts(source, 'source').map((dim) => S.normalizeDim(dim, shape.length))
    const to = toInts(destination, 'destination').map((dim) => S.normalizeDim(dim, shape.length))
    const order: (number | null)[] = Array(shape.length).fill(null)
    from.forEach((axis, index) => { order[to[index]] = axis })
    const rest = shape.map((_, index) => index).filter((axis) => !from.includes(axis))
    return order.map((axis) => shape[axis ?? rest.shift()!])
  }

  private chunk(tensor: Tensor, chunks: number, dim: number): PyTuple {
    const axis = S.normalizeDim(dim, tensor.shape.length)
    const size = Math.ceil(tensor.shape[axis] / chunks)
    const parts: Tensor[] = []
    for (let start = 0; start < tensor.shape[axis]; start += size) {
      const shape = [...tensor.shape]
      shape[axis] = Math.min(size, tensor.shape[axis] - start)
      parts.push(new Tensor(shape))
    }
    return this.recordOp('chunk', 'split', [tensor], new PyTuple(parts), this.location.line, { chunks, dim }) as PyTuple
  }

  private split(tensor: Tensor, sizes: Value, dim: number): PyTuple {
    const axis = S.normalizeDim(dim, tensor.shape.length)
    const total = tensor.shape[axis]
    const lengths = sizes instanceof PyList || sizes instanceof PyTuple ? toInts(sizes, 'split_size') : Array.from({ length: Math.ceil(total / toInt(sizes, 'split_size')) }, (_, index) => Math.min(toInt(sizes, 'split_size'), total - index * toInt(sizes, 'split_size')))
    if (lengths.reduce((sum, value) => sum + value, 0) !== total) throw this.error(`split sizes ${lengths.join('+')} do not add up to ${total}`)
    return this.recordOp('split', 'split', [tensor], new PyTuple(lengths.map((length) => { const shape = [...tensor.shape]; shape[axis] = length; return new Tensor(shape) })), this.location.line, { dim }) as PyTuple
  }

  private reduction(name: string, tensor: Tensor, dim: Value, keepdim: boolean): Value {
    const dims = dim === null ? null : dim instanceof PyTuple || dim instanceof PyList ? toInts(dim, 'dim') : toInt(dim, 'dim')
    return this.recordOp(name, 'pool', [tensor], new Tensor(S.reduce(tensor.shape, dims, keepdim)), this.location.line, { dim: dims === null ? 'all' : Array.isArray(dims) ? dims : dims, keepdim })
  }

  private create(op: string, shape: number[], dtype: Tensor['dtype'], line: number): Tensor {
    const tensor = new Tensor(shape, null, dtype)
    if (this.recording) return this.recordOp(op, 'tensor', [], tensor, line, undefined, undefined, 0, `torch.${op}`) as Tensor
    return tensor
  }

  private tensorFunction(name: string, args: Value[], _kwargs: Map<string, Value>): Value {
    const line = this.location.line
    const first = args[0]
    const t = (value: Value, what = name): Tensor => {
      if (!(value instanceof Tensor)) throw this.error(`${what} expects a Tensor, got ${formatValue(value)}`)
      return value
    }
    switch (name) {
      case 'maximum': case 'minimum': case 'where': {
        const tensors = args.filter((item): item is Tensor => item instanceof Tensor)
        const shape = tensors.reduce((acc, tensor) => S.broadcast(acc, tensor.shape, name), [] as number[])
        return this.recordOp(name, 'generic', tensors, new Tensor(shape), line)
      }
    }
    return t(first)
  }

  // ───────────────────────────── builtins ─────────────────────────────

  private installBuiltins(): void {
    const b = this.builtins.vars
    const fn = (name: string, impl: NativeImpl) => { const native = new NativeFn(name, impl); b.set(name, native); return native }
    const typeFn = (name: string, impl: NativeImpl) => fn(name, impl)
    typeFn('int', (args) => args.length ? (typeof args[0] === 'string' ? parseInt(args[0], 10) : Math.trunc(toNumber(args[0], 'int'))) : 0)
    typeFn('float', (args) => args.length ? (typeof args[0] === 'string' ? (args[0] === 'inf' ? Infinity : args[0] === '-inf' ? -Infinity : parseFloat(args[0])) : toNumber(args[0], 'float')) : 0)
    typeFn('str', (args) => args.length ? pyStr(args[0]) : '')
    typeFn('bool', (args) => args.length ? isTruthy(args[0]) : false)
    typeFn('list', (args) => new PyList(args.length ? this.iterate(args[0]) : []))
    typeFn('tuple', (args) => new PyTuple(args.length ? this.iterate(args[0]) : []))
    typeFn('set', (args) => new PySet(args.length ? this.iterate(args[0]) : []))
    typeFn('dict', (args, kwargs) => {
      const dict = new PyDict()
      if (args[0] instanceof PyDict) args[0].map.forEach((value, key) => dict.map.set(key, value))
      else if (args[0]) this.iterate(args[0]).forEach((pair) => { const [key, value] = this.iterate(pair); dict.map.set(this.key(key), value) })
      kwargs.forEach((value, key) => dict.map.set(key, value))
      return dict
    })
    fn('range', (args) => {
      const values = args.map((arg) => toInt(arg, 'range'))
      if (values.length === 1) return new PyRange(0, values[0], 1)
      return new PyRange(values[0], values[1], values[2] ?? 1)
    })
    fn('len', (args) => {
      const value = args[0]
      if (value instanceof Tensor) return value.shape[0] ?? 0
      if (value instanceof ModuleObj) {
        const len = value.kind === 'user' ? this.bindFrom(value.cls!, '__len__', value) : undefined
        return len !== undefined ? toInt(this.call(len, [], new Map(), this.location), '__len__') : value.children.size
      }
      if (value instanceof PyDict) return value.map.size
      if (this.atlasKind(value) === 'DataLoader') { const loader = value as PyInstance; const size = this.datasetSize(loader.attrs.get('dataset') ?? null); const batch = toInt(loader.attrs.get('batch_size') ?? 1, 'batch_size'); return loader.attrs.get('drop_last') === true ? Math.floor(size / batch) : Math.ceil(size / batch) }
      if (value instanceof PyInstance) return this.datasetSize(value)
      return this.iterate(value).length
    })
    const extreme = (sign: number) => (args: Value[], kwargs: Map<string, Value>, site: Site) => {
      const items = args.length === 1 ? this.iterate(args[0]) : args
      if (!items.length) { if (kwargs.has('default')) return kwargs.get('default')!; throw this.error('ValueError: max()/min() of an empty sequence') }
      const key = kwargs.get('key') ?? null
      const score = (item: Value) => key === null ? item : this.call(key, [item], new Map(), site)
      return items.reduce((best, item) => sign * this.compareValues(score(item), score(best)) > 0 ? item : best)
    }
    fn('max', extreme(1))
    fn('min', extreme(-1))
    fn('sum', (args) => this.iterate(args[0]).reduce((total: Value, item) => this.binary('+', total, item, this.location.line), args[1] ?? 0))
    fn('abs', (args) => args[0] instanceof Tensor ? args[0] : Math.abs(toNumber(args[0], 'abs')))
    fn('round', (args) => args.length > 1 ? Number(toNumber(args[0], 'round').toFixed(toInt(args[1], 'ndigits'))) : Math.round(toNumber(args[0], 'round')))
    fn('pow', (args) => toNumber(args[0], 'pow') ** toNumber(args[1], 'pow'))
    fn('divmod', (args) => tupleOf([Math.floor(toNumber(args[0], 'a') / toNumber(args[1], 'b')), toNumber(args[0], 'a') % toNumber(args[1], 'b')]))
    fn('zip', (args) => {
      const lists = args.map((arg) => this.iterate(arg))
      const length = Math.min(...lists.map((list) => list.length))
      return new PyList(Array.from({ length }, (_, index) => tupleOf(lists.map((list) => list[index]))))
    })
    fn('enumerate', (args, kwargs) => new PyList(this.iterate(args[0]).map((item, index) => tupleOf([index + toInt(args[1] ?? kwargs.get('start') ?? 0, 'start'), item]))))
    fn('reversed', (args) => new PyList(this.iterate(args[0]).reverse()))
    fn('sorted', (args, kwargs, site) => {
      const key = kwargs.get('key') ?? null
      const keyed = this.iterate(args[0]).map((item) => [key === null ? item : this.call(key, [item], new Map(), site), item] as const)
      keyed.sort((a, c) => this.compareValues(a[0], c[0]))
      if (isTruthy(kwargs.get('reverse') ?? false)) keyed.reverse()
      return new PyList(keyed.map(([, item]) => item))
    })
    fn('format', (args) => this.pyFormat(args[0], args.length > 1 ? String(args[1]) : ''))
    fn('any', (args) => this.iterate(args[0]).some(isTruthy))
    fn('all', (args) => this.iterate(args[0]).every(isTruthy))
    fn('map', (args, _, site) => new PyList(this.iterate(args[1]).map((item) => this.call(args[0], [item], new Map(), site))))
    fn('filter', (args, _, site) => new PyList(this.iterate(args[1]).filter((item) => isTruthy(args[0] === null ? item : this.call(args[0], [item], new Map(), site)))))
    fn('iter', (args) => new PyList(this.iterate(args[0])))
    fn('next', (args) => this.iterate(args[0])[0] ?? null)
    fn('print', (args, kwargs) => { if (this.logs.length < 200) this.logs.push(args.map((arg) => pyStr(arg)).join(typeof kwargs.get('sep') === 'string' ? kwargs.get('sep') as string : ' ')); return null })
    fn('callable', (args) => args[0] instanceof NativeFn || args[0] instanceof PyFunction || args[0] instanceof BoundMethod || args[0] instanceof PyClass || args[0] instanceof ModuleObj)
    fn('repr', (args) => pyRepr(args[0]))
    fn('type', (args) => args[0] instanceof PyInstance ? args[0].cls : typeName(args[0]))
    fn('id', () => 0)
    fn('hasattr', (args) => { try { this.getAttr(args[0], String(args[1])); return true } catch { return false } })
    fn('getattr', (args) => { try { return this.getAttr(args[0], String(args[1])) } catch (error) { if (args.length > 2) return args[2]; throw error } })
    fn('setattr', (args) => { this.setAttr(args[0], String(args[1]), args[2]); return null })
    fn('isinstance', (args) => {
      const value = args[0]
      const types = args[1] instanceof PyTuple ? args[1].items : [args[1]]
      return types.some((type) => {
        if (type instanceof NativeFn) {
          if (type.name === 'Tensor') return value instanceof Tensor
          if (type.name === 'int') return typeof value === 'number' && Number.isInteger(value) && typeof value !== 'boolean'
          if (type.name === 'float') return typeof value === 'number'
          if (type.name === 'Module') return value instanceof ModuleObj
          if (isLayer(type.name)) return value instanceof ModuleObj && value.className === type.name
          return typeName(value) === type.name
        }
        if (type === NN_MODULE) return value instanceof ModuleObj
        if (type instanceof PyClass) {
          const cls = value instanceof ModuleObj ? value.cls : value instanceof PyInstance ? value.cls : undefined
          const inherits = (candidate?: PyClass): boolean => Boolean(candidate && (candidate === type || candidate.bases.some((base) => base instanceof PyClass && inherits(base))))
          return inherits(cls)
        }
        return false
      })
    })
    fn('super', () => {
      const frame = this.frames[this.frames.length - 1]
      if (!frame?.owner) throw this.error('super() is only supported inside methods')
      return new SuperProxy(frame.self ?? null, frame.owner)
    })
    for (const name of ['Exception', 'ValueError', 'TypeError', 'RuntimeError', 'NotImplementedError', 'AssertionError', 'KeyError', 'IndexError']) {
      fn(name, (args) => args.length ? formatValue(args[0]) : name)
    }
    b.set('object', null)
    b.set('NotImplemented', null)
    b.set('__name__', '__main__')

    this.installTorch()
  }

  private installTorch(): void {
    const b = this.builtins.vars
    const ns = (name: string, entries: [string, Value][], permissive = false) => new Namespace(name, new Map(entries), permissive)
    const native = (name: string, impl: NativeImpl): [string, Value] => [name, new NativeFn(name, impl)]
    const line = () => this.location.line
    const dtypeOf = (kwargs: Map<string, Value>, fallback: Tensor['dtype']) => {
      const dtype = kwargs.get('dtype')
      return dtype instanceof Dtype ? dtype.kind : fallback
    }
    const t = (value: Value | undefined, what: string): Tensor => {
      if (!(value instanceof Tensor)) throw this.error(`${what} expects a Tensor, got ${value === undefined ? 'nothing' : formatValue(value)}`)
      return value
    }
    const activationFn = (name: string, type: ModelNodeType = 'activation'): [string, Value] => native(name, (args, kwargs) => {
      const x = args[0] ?? kwargs.get('input')
      if (typeof x === 'number') {
        const math: Record<string, (value: number) => number> = { exp: Math.exp, log: Math.log, sqrt: Math.sqrt, sin: Math.sin, cos: Math.cos, tanh: Math.tanh, sigmoid: (value) => 1 / (1 + Math.exp(-value)), relu: (value) => Math.max(0, value), rsqrt: (value) => 1 / Math.sqrt(value) }
        return math[name]?.(x) ?? x
      }
      const tensor = t(x, name)
      return this.recordOp(name, type, [tensor], new Tensor(tensor.shape), line())
    })
    const layers = supportedLayers
      .map((className): [string, Value] => [className, new NativeFn(className, (args, kwargs, site) => {
        if (className === 'Identity') { args = []; kwargs = new Map() }
        const layer = createLayer(className, args, kwargs)
        layer.params.forEach((tensor, key) => { tensor.label = key })
        return new ModuleObj('layer', className, site, undefined, layer)
      })])
    const container = (kind: 'sequential' | 'modulelist' | 'moduledict', className: string): [string, Value] => [className, new NativeFn(className, (args, _, site) => {
      const module = new ModuleObj(kind, className, site)
      if (kind === 'sequential') {
        if (args.length === 1 && args[0] instanceof PyDict) args[0].map.forEach((value, key) => { if (value instanceof ModuleObj) module.addChild(String(key), value) })
        else args.forEach((arg, index) => { if (!(arg instanceof ModuleObj)) throw this.error(`nn.Sequential argument ${index} is not a module (${formatValue(arg)})`); module.addChild(String(index), arg) })
      } else if (kind === 'modulelist') {
        if (args[0]) this.iterate(args[0]).forEach((item, index) => { if (!(item instanceof ModuleObj)) throw this.error(`nn.ModuleList item ${index} is not a module`); module.addChild(String(index), item) })
      } else if (args[0] instanceof PyDict) args[0].map.forEach((value, key) => { if (value instanceof ModuleObj) module.addChild(String(key), value) })
      return module
    })]
    const parameter = native('Parameter', (args) => {
      const source = t(args[0], 'nn.Parameter')
      return new Tensor(source.shape, null, source.dtype, 'parameter')
    })

    // Training: optimizers, schedulers, data loading and mixed precision. They draw one training step.
    const builtinSite = { file: '<torch>', line: 0, endLine: 0 }
    const object = (name: string, kind: string, attrs: [string, Value][] = []): PyInstance => {
      const instance = new PyInstance(new PyClass(name, [], builtinSite))
      instance.attrs.set('__atlas__', kind)
      attrs.forEach(([key, value]) => instance.attrs.set(key, value))
      return instance
    }
    const setting = (value: Value) => typeof value === 'number' || typeof value === 'boolean' || typeof value === 'string' || value instanceof PyTuple
    const optimizerFn = (name: string, lr: number): [string, Value] => native(name, (args, kwargs) => {
      const params = args[0] ?? kwargs.get('params') ?? new PyList([])
      const settings: [string, Value][] = [['lr', args[1] ?? kwargs.get('lr') ?? lr]]
      kwargs.forEach((value, key) => { if (key !== 'params' && key !== 'lr' && setting(value)) settings.push([key, value]) })
      settings.push(['parameters', tensorsIn(params).reduce((sum, tensor) => sum + tensor.numel, 0)])
      const optimizer = object(name, 'Optimizer', settings)
      optimizer.attrs.set('param_groups', new PyList([new PyDict([['lr', settings[0][1]]])]))
      optimizer.attrs.set('step', new NativeFn(`${name}.step`, () => this.optimizerStep(optimizer)))
      optimizer.attrs.set('zero_grad', new NativeFn(`${name}.zero_grad`, () => { if (this.training !== 'done') optimizer.attrs.set('__zero_grad__', true); return null }))
      optimizer.attrs.set('state_dict', new NativeFn('state_dict', () => new PyDict()))
      optimizer.attrs.set('load_state_dict', noop('load_state_dict'))
      optimizer.attrs.set('add_param_group', noop('add_param_group'))
      this.openTraining()
      return optimizer
    })
    const schedulerFn = (name: string): [string, Value] => native(name, (args, kwargs) => {
      const optimizer = args[0] ?? kwargs.get('optimizer') ?? null
      if (optimizer instanceof PyInstance && name !== 'SequentialLR' && name !== 'ChainedScheduler') optimizer.attrs.set('__scheduler__', name)
      const lr = optimizer instanceof PyInstance ? optimizer.attrs.get('lr') ?? 0 : 0
      return object(name, 'Scheduler', [['step', noop(`${name}.step`)], ['get_last_lr', new NativeFn('get_last_lr', () => new PyList([lr]))], ['state_dict', new NativeFn('state_dict', () => new PyDict())], ['load_state_dict', noop('load_state_dict')]])
    })
    const optim = ns('torch.optim', [
      optimizerFn('SGD', 0.001), optimizerFn('Adam', 0.001), optimizerFn('AdamW', 0.001), optimizerFn('RMSprop', 0.01), optimizerFn('Adagrad', 0.01),
      optimizerFn('Adadelta', 1), optimizerFn('NAdam', 0.002), optimizerFn('RAdam', 0.001), optimizerFn('Adamax', 0.002),
      ['Optimizer', new NativeFn('Optimizer', () => null)],
      ['lr_scheduler', ns('torch.optim.lr_scheduler', ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'LambdaLR', 'MultiplicativeLR', 'OneCycleLR', 'CyclicLR', 'LinearLR', 'ConstantLR', 'PolynomialLR', 'ReduceLROnPlateau', 'SequentialLR', 'ChainedScheduler'].map(schedulerFn))],
    ])
    const dataset = (name: string, size: number, source: Value, attrs: [string, Value][] = []) => object(name, name === 'TensorDataset' ? 'TensorDataset' : 'Subset', [['__size__', size], ['dataset', source], ...attrs])
    const data = ns('torch.utils.data', [
      ['Dataset', new NativeFn('Dataset', () => null)], ['IterableDataset', new NativeFn('IterableDataset', () => null)],
      native('TensorDataset', (args) => {
        const tensors = args.map((arg, index) => t(arg, `TensorDataset[${index}]`))
        if (tensors.some((tensor) => tensor.shape[0] !== tensors[0].shape[0])) throw this.error(`TensorDataset: tensors have different first dimensions (${tensors.map((tensor) => tensor.shape[0]).join(', ')})`)
        return dataset('TensorDataset', tensors[0]?.shape[0] ?? 0, null, [['tensors', new PyTuple(tensors)]])
      }),
      native('Subset', (args, kwargs) => dataset('Subset', this.iterate(args[1] ?? kwargs.get('indices') ?? new PyList([])).length, args[0] ?? kwargs.get('dataset') ?? null)),
      native('random_split', (args, kwargs) => {
        const source = args[0] ?? kwargs.get('dataset') ?? null
        const total = this.datasetSize(source)
        const lengths = this.iterate(args[1] ?? kwargs.get('lengths') ?? new PyList([])).map((value) => toNumber(value, 'lengths'))
        const sizes = lengths.every((value) => value <= 1) && lengths.some((value) => !Number.isInteger(value)) ? lengths.map((value) => Math.floor(value * total)) : lengths
        return new PyList(sizes.map((size) => dataset('Subset', size, source)))
      }),
      native('DataLoader', (args, kwargs) => {
        const source = args[0] ?? kwargs.get('dataset') ?? null
        return object('DataLoader', 'DataLoader', [['dataset', source], ['batch_size', args[1] ?? kwargs.get('batch_size') ?? 1], ['drop_last', kwargs.get('drop_last') ?? false]])
      }),
      native('default_collate', (args) => this.collate(args[0] ?? null, 1)),
    ], true)
    const gradScaler = native('GradScaler', () => object('GradScaler', 'GradScaler', [
      ['scale', new NativeFn('scale', (args) => args[0] ?? null)],
      ['step', new NativeFn('step', (args) => args[0] instanceof PyInstance ? this.optimizerStep(args[0]) : null)],
      ['update', noop('update')], ['unscale_', noop('unscale_')],
    ]))
    const amp = ns('torch.amp', [gradScaler, native('autocast', () => new ContextManager('autocast'))])
    const clip = (name: string, key: string): [string, Value] => native(name, (args, kwargs) => {
      const limit = args[1] ?? kwargs.get(key) ?? null
      if (!this.recording || !this.gradient) return new Tensor([])
      const grad = this.recordOp(name.replace(/_$/, ''), 'optimizer', [this.gradient], new Tensor([]), line(), typeof limit === 'number' ? { [key]: limit } : undefined, undefined, 0, `nn.utils.${name}`) as Tensor
      this.gradient = grad
      return grad
    })

    const F = ns('torch.nn.functional', [
      ...['relu', 'relu6', 'gelu', 'silu', 'mish', 'elu', 'leaky_relu', 'sigmoid', 'tanh', 'softplus', 'hardswish', 'softmax', 'log_softmax'].map((name) => activationFn(name)),
      native('dropout', (args) => args[0] ?? null),
      native('normalize', (args) => { const x = t(args[0], 'F.normalize'); return this.passThrough(x, x.shape) }),
      native('pad', (args, kwargs) => {
        const x = t(args[0], 'F.pad')
        const pad = toInts(args[1] ?? kwargs.get('pad')!, 'pad')
        const shape = [...x.shape]
        for (let index = 0; index < pad.length / 2; index++) shape[shape.length - 1 - index] += pad[2 * index] + pad[2 * index + 1]
        return this.passThrough(x, shape, x.dtype, 'pad')
      }),
      native('layer_norm', (args) => { const x = t(args[0], 'F.layer_norm'); return this.recordOp('layer_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      native('rms_norm', (args) => { const x = t(args[0], 'F.rms_norm'); return this.recordOp('rms_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      native('group_norm', (args) => { const x = t(args[0], 'F.group_norm'); return this.recordOp('group_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      native('linear', (args, kwargs) => {
        const x = t(args[0], 'F.linear')
        const weight = t(args[1] ?? kwargs.get('weight'), 'F.linear weight')
        if (x.shape[x.shape.length - 1] !== weight.shape[1]) throw this.error(`F.linear: input features ${x.shape[x.shape.length - 1]} do not match weight ${S.fmt(weight.shape)}`)
        return this.recordOp('linear', 'linear', [x, weight], new Tensor([...x.shape.slice(0, -1), weight.shape[0]]), line())
      }),
      native('scaled_dot_product_attention', (args, kwargs) => {
        const q = t(args[0] ?? kwargs.get('query'), 'scaled_dot_product_attention')
        const k = t(args[1] ?? kwargs.get('key'), 'scaled_dot_product_attention')
        const v = t(args[2] ?? kwargs.get('value'), 'scaled_dot_product_attention')
        if (q.shape[q.shape.length - 1] !== k.shape[k.shape.length - 1]) throw this.error(`Attention query dim ${q.shape[q.shape.length - 1]} ≠ key dim ${k.shape[k.shape.length - 1]}`)
        const config = { is_causal: Boolean(kwargs.get('is_causal') ?? false), heads: q.shape.length === 4 ? q.shape[1] : 1, head_dim: q.shape[q.shape.length - 1] }
        return this.recordOp('attention', 'attention', [q, k, v], new Tensor([...q.shape.slice(0, -1), v.shape[v.shape.length - 1]]), line(), config, undefined, 0, 'scaled_dot_product_attention')
      }),
      native('interpolate', (args, kwargs) => {
        const x = t(args[0], 'F.interpolate')
        return this.recordOp('interpolate', 'upsample', [x], new Tensor(interpolateShape(x.shape, args[1] ?? kwargs.get('size') ?? null, args[2] ?? kwargs.get('scale_factor') ?? null)), line(), { mode: String(kwargs.get('mode') ?? 'nearest') })
      }),
      ...(['max_pool1d', 'max_pool2d', 'avg_pool1d', 'avg_pool2d'] as const).map((name) => native(name, (args, kwargs) => {
        const x = t(args[0], `F.${name}`)
        const dims = name.endsWith('1d') ? 1 : 2
        const kernel = nTuple(args[1] ?? kwargs.get('kernel_size')!, dims, 'kernel_size')
        const strideValue = args[2] ?? kwargs.get('stride') ?? null
        const stride = strideValue === null ? kernel : nTuple(strideValue, dims, 'stride')
        const padding = nTuple(args[3] ?? kwargs.get('padding') ?? 0, dims, 'padding')
        const shape = [...x.shape.slice(0, -dims), ...x.shape.slice(-dims).map((size, index) => S.convAxis(size, kernel[index], stride[index], padding[index]))]
        return this.recordOp(name, 'pool', [x], new Tensor(shape), line(), { kernel_size: kernel[0], stride: stride[0] })
      })),
      ...(['adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_max_pool2d'] as const).map((name) => native(name, (args, kwargs) => {
        const x = t(args[0], `F.${name}`)
        const dims = name.endsWith('1d') ? 1 : 2
        const size = nTuple(args[1] ?? kwargs.get('output_size')!, dims, 'output_size')
        return this.recordOp(name, 'pool', [x], new Tensor([...x.shape.slice(0, -dims), ...size]), line(), { output_size: size[0] })
      })),
      ...['cross_entropy', 'mse_loss', 'l1_loss', 'smooth_l1_loss', 'huber_loss', 'nll_loss', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'kl_div', 'cosine_embedding_loss',
        'margin_ranking_loss', 'triplet_margin_loss', 'soft_margin_loss', 'hinge_embedding_loss', 'poisson_nll_loss', 'gaussian_nll_loss', 'multilabel_soft_margin_loss'].map((name) => native(name, (args, kwargs) => {
        const inputs = [...args, kwargs.get('input'), kwargs.get('target')].filter((item): item is Tensor => item instanceof Tensor).slice(0, name === 'triplet_margin_loss' || name === 'margin_ranking_loss' || name === 'cosine_embedding_loss' || name === 'gaussian_nll_loss' ? 3 : 2)
        const reduction = kwargs.get('reduction') ?? (kwargs.get('reduce') === false ? 'none' : 'mean')
        return this.recordOp(name, 'loss', inputs, new Tensor(lossShape(name, inputs, reduction)), line(), reduction !== 'mean' ? { reduction: String(reduction) } : undefined)
      })),
      native('cosine_similarity', (args, kwargs) => {
        const [a, c] = [t(args[0] ?? kwargs.get('x1'), 'F.cosine_similarity'), t(args[1] ?? kwargs.get('x2'), 'F.cosine_similarity')]
        const shape = S.broadcast(a.shape, c.shape, 'cosine_similarity')
        const axis = S.normalizeDim(toInt(args[2] ?? kwargs.get('dim') ?? 1, 'dim'), shape.length)
        return this.recordOp('cosine_similarity', 'generic', [a, c], new Tensor(shape.filter((_, index) => index !== axis)), line())
      }),
      native('pairwise_distance', (args, kwargs) => {
        const [a, c] = [t(args[0], 'F.pairwise_distance'), t(args[1], 'F.pairwise_distance')]
        const shape = S.broadcast(a.shape, c.shape, 'pairwise_distance')
        return this.recordOp('pairwise_distance', 'generic', [a, c], new Tensor(isTruthy(kwargs.get('keepdim') ?? false) ? [...shape.slice(0, -1), 1] : shape.slice(0, -1)), line())
      }),
      ...['softmin', 'logsigmoid', 'hardtanh', 'selu', 'celu', 'softsign', 'tanhshrink', 'threshold', 'hardsigmoid', 'hardshrink', 'softshrink', 'relu_', 'elu_', 'leaky_relu_', 'rrelu', 'prelu', 'gumbel_softmax', 'silu_'].map((name) => activationFn(name)),
      ...['dropout1d', 'dropout2d', 'dropout3d', 'alpha_dropout', 'feature_alpha_dropout'].map((name) => native(name, (args) => args[0] ?? null)),
      native('batch_norm', (args) => { const x = t(args[0], 'F.batch_norm'); return this.recordOp('batch_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      native('instance_norm', (args) => { const x = t(args[0], 'F.instance_norm'); return this.recordOp('instance_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      native('local_response_norm', (args) => { const x = t(args[0], 'F.local_response_norm'); return this.recordOp('local_response_norm', 'norm', [x], new Tensor(x.shape), line()) }),
      ...([1, 2, 3] as const).map((dims) => native(`conv${dims}d`, (args, kwargs) => {
        const x = t(args[0] ?? kwargs.get('input'), `F.conv${dims}d`)
        const weight = t(args[1] ?? kwargs.get('weight'), `F.conv${dims}d weight`)
        const stride = nTuple(args[3] ?? kwargs.get('stride') ?? 1, dims, 'stride')
        const paddingValue = args[4] ?? kwargs.get('padding') ?? 0
        const dilation = nTuple(args[5] ?? kwargs.get('dilation') ?? 1, dims, 'dilation')
        const groups = toInt(args[6] ?? kwargs.get('groups') ?? 1, 'groups')
        const channelAxis = x.shape.length - dims - 1
        if (x.shape[channelAxis] !== weight.shape[1] * groups) throw this.error(`F.conv${dims}d: input has ${x.shape[channelAxis]} channels but the weight expects ${weight.shape[1] * groups}`)
        const padding = typeof paddingValue === 'string' ? null : nTuple(paddingValue, dims, 'padding')
        const spatial = x.shape.slice(channelAxis + 1).map((size, index) => padding === null ? (paddingValue === 'same' ? size : S.convAxis(size, weight.shape[2 + index], stride[index], 0, dilation[index])) : S.convAxis(size, weight.shape[2 + index], stride[index], padding[index], dilation[index]))
        return this.recordOp(`conv${dims}d`, dims === 1 ? 'conv1d' : dims === 2 ? 'conv2d' : 'conv3d', [x, weight], new Tensor([...x.shape.slice(0, channelAxis), weight.shape[0], ...spatial]), line())
      })),
      ...([1, 2, 3] as const).map((dims) => native(`conv_transpose${dims}d`, (args, kwargs) => {
        const x = t(args[0], `F.conv_transpose${dims}d`)
        const weight = t(args[1] ?? kwargs.get('weight'), `F.conv_transpose${dims}d weight`)
        const stride = nTuple(args[3] ?? kwargs.get('stride') ?? 1, dims, 'stride')
        const padding = nTuple(args[4] ?? kwargs.get('padding') ?? 0, dims, 'padding')
        const outputPadding = nTuple(args[5] ?? kwargs.get('output_padding') ?? 0, dims, 'output_padding')
        const groups = toInt(args[6] ?? kwargs.get('groups') ?? 1, 'groups')
        const spatial = x.shape.slice(-dims).map((size, index) => (size - 1) * stride[index] - 2 * padding[index] + weight.shape[2 + index] + outputPadding[index])
        return this.recordOp(`conv_transpose${dims}d`, 'upsample', [x, weight], new Tensor([...x.shape.slice(0, -dims - 1), weight.shape[1] * groups, ...spatial]), line())
      })),
      native('unfold', (args, kwargs) => {
        const x = t(args[0], 'F.unfold')
        const kernel = nTuple(args[1] ?? kwargs.get('kernel_size')!, 2, 'kernel_size')
        const dilation = nTuple(args[2] ?? kwargs.get('dilation') ?? 1, 2, 'dilation')
        const padding = nTuple(args[3] ?? kwargs.get('padding') ?? 0, 2, 'padding')
        const stride = nTuple(args[4] ?? kwargs.get('stride') ?? 1, 2, 'stride')
        const [h, w] = x.shape.slice(-2)
        const blocks = S.convAxis(h, kernel[0], stride[0], padding[0], dilation[0]) * S.convAxis(w, kernel[1], stride[1], padding[1], dilation[1])
        return this.recordOp('unfold', 'reshape', [x], new Tensor([...x.shape.slice(0, -3), x.shape[x.shape.length - 3] * kernel[0] * kernel[1], blocks]), line())
      }),
      native('fold', (args, kwargs) => {
        const x = t(args[0], 'F.fold')
        const size = nTuple(args[1] ?? kwargs.get('output_size')!, 2, 'output_size')
        const kernel = nTuple(args[2] ?? kwargs.get('kernel_size')!, 2, 'kernel_size')
        return this.recordOp('fold', 'reshape', [x], new Tensor([...x.shape.slice(0, -2), x.shape[x.shape.length - 2] / (kernel[0] * kernel[1]), ...size]), line())
      }),
      native('grid_sample', (args) => {
        const [x, grid] = [t(args[0], 'F.grid_sample'), t(args[1], 'F.grid_sample grid')]
        return this.recordOp('grid_sample', 'upsample', [x, grid], new Tensor([x.shape[0], x.shape[1], ...grid.shape.slice(1, -1)]), line())
      }),
      native('affine_grid', (args, kwargs) => {
        const theta = t(args[0], 'F.affine_grid')
        const size = toInts(args[1] ?? kwargs.get('size')!, 'size')
        return this.recordOp('affine_grid', 'tensor', [theta], new Tensor([size[0], ...size.slice(2), size.length - 2]), line())
      }),
      native('pixel_unshuffle', (args) => {
        const x = t(args[0], 'F.pixel_unshuffle')
        const r = toInt(args[1], 'downscale_factor')
        const [c, h, w] = x.shape.slice(-3)
        return this.recordOp('pixel_unshuffle', 'downsample', [x], new Tensor([...x.shape.slice(0, -3), c * r * r, h / r, w / r]), line())
      }),
      ...(['max_pool3d', 'avg_pool3d'] as const).map((name) => native(name, (args, kwargs) => {
        const x = t(args[0], `F.${name}`)
        const kernel = nTuple(args[1] ?? kwargs.get('kernel_size')!, 3, 'kernel_size')
        const strideValue = args[2] ?? kwargs.get('stride') ?? null
        const stride = strideValue === null ? kernel : nTuple(strideValue, 3, 'stride')
        const padding = nTuple(args[3] ?? kwargs.get('padding') ?? 0, 3, 'padding')
        return this.recordOp(name, 'pool', [x], new Tensor([...x.shape.slice(0, -3), ...x.shape.slice(-3).map((size, index) => S.convAxis(size, kernel[index], stride[index], padding[index]))]), line())
      })),
      ...(['adaptive_avg_pool3d', 'adaptive_max_pool1d', 'adaptive_max_pool3d'] as const).map((name) => native(name, (args, kwargs) => {
        const x = t(args[0], `F.${name}`)
        const dims = name.endsWith('1d') ? 1 : 3
        const size = nTuple(args[1] ?? kwargs.get('output_size')!, dims, 'output_size')
        return this.recordOp(name, 'pool', [x], new Tensor([...x.shape.slice(0, -dims), ...size]), line())
      })),
      native('one_hot', (args, kwargs) => { const x = t(args[0], 'F.one_hot'); return this.recordOp('one_hot', 'embedding', [x], new Tensor([...x.shape, toInt(args[1] ?? kwargs.get('num_classes')!, 'num_classes')]), line()) }),
      native('embedding', (args) => { const x = t(args[0], 'F.embedding'); const weight = t(args[1], 'F.embedding'); return this.recordOp('embedding', 'embedding', [x, weight], new Tensor([...x.shape, weight.shape[1]]), line()) }),
      native('glu', (args, kwargs) => {
        const x = t(args[0], 'F.glu')
        const axis = S.normalizeDim(toInt(args[1] ?? kwargs.get('dim') ?? -1, 'dim'), x.shape.length)
        const shape = [...x.shape]
        shape[axis] /= 2
        return this.recordOp('glu', 'activation', [x], new Tensor(shape), line())
      }),
      native('pixel_shuffle', (args) => {
        const x = t(args[0], 'F.pixel_shuffle')
        const r = toInt(args[1], 'upscale_factor')
        const [c, h, w] = x.shape.slice(-3)
        return this.recordOp('pixel_shuffle', 'upsample', [x], new Tensor([...x.shape.slice(0, -3), c / (r * r), h * r, w * r]), line())
      }),
    ])

    const nn = ns('torch.nn', [
      ['Module', NN_MODULE], parameter, ...layers,
      container('sequential', 'Sequential'), container('modulelist', 'ModuleList'), container('moduledict', 'ModuleDict'),
      ['functional', F], ['init', ns('torch.nn.init', [], true)], ['utils', ns('torch.nn.utils', [clip('clip_grad_norm_', 'max_norm'), clip('clip_grad_value_', 'clip_value')], true)],
    ])

    const creation = (name: string, dtype: Tensor['dtype'] = 'float'): [string, Value] => native(name, (args, kwargs) => {
      const sizeArgs = args.filter((arg) => !(arg instanceof Dtype))
      const shape = sizeArgs.length ? shapeArgs(sizeArgs, 'size') : toInts(kwargs.get('size') ?? new PyTuple([]), 'size')
      return this.create(name, shape, dtypeOf(kwargs, dtype), line())
    })
    const like = (name: string, dtype?: Tensor['dtype']): [string, Value] => native(name, (args, kwargs) => {
      const source = t(args[0], name)
      return this.recordOp(name, name.startsWith('rand') ? 'sampling' : 'tensor', this.recording ? [] : [source], new Tensor(source.shape, null, dtypeOf(kwargs, dtype ?? source.dtype)), line(), undefined, undefined, 0, `torch.${name}`)
    })
    const passFn = (name: string, method: string): [string, Value] => native(name, (args, kwargs, site) => {
      const tensor = t(args[0], `torch.${name}`)
      return this.call(this.tensorAttr(tensor, method), args.slice(1), kwargs, site)
    })
    const dtypes: [string, Value][] = [
      ['float32', new Dtype('torch.float32', 'float')], ['float', new Dtype('torch.float32', 'float')], ['float16', new Dtype('torch.float16', 'float')], ['half', new Dtype('torch.float16', 'float')],
      ['bfloat16', new Dtype('torch.bfloat16', 'float')], ['float64', new Dtype('torch.float64', 'float')], ['long', new Dtype('torch.int64', 'long')], ['int64', new Dtype('torch.int64', 'long')],
      ['int32', new Dtype('torch.int32', 'long')], ['int', new Dtype('torch.int32', 'long')], ['bool', new Dtype('torch.bool', 'bool')],
    ]

    const torch = ns('torch', [
      ['nn', nn], ...dtypes,
      ['Tensor', new NativeFn('Tensor', () => { throw this.error('Use torch.randn(...) to create example tensors') })],
      ['Size', new NativeFn('Size', (args) => new PyTuple(args.length ? this.iterate(args[0]) : []))],
      ['cuda', ns('torch.cuda', [native('is_available', () => false), native('device_count', () => 0), ['amp', amp]], true)],
      ['optim', optim], ['amp', amp], ['utils', ns('torch.utils', [['data', data]], true)],
      native('save', () => null), native('load', () => new PyDict()), native('compile', (args) => args[0] ?? null), native('set_float32_matmul_precision', () => null),
      ['backends', ns('torch.backends', [], true)],
      native('device', (args) => String(args[0] ?? 'cpu')),
      native('manual_seed', () => null), native('set_grad_enabled', () => new ContextManager('grad')),
      native('no_grad', () => new ContextManager('no_grad')), native('inference_mode', () => new ContextManager('inference_mode')), native('autocast', () => new ContextManager('autocast')),
      native('is_tensor', (args) => args[0] instanceof Tensor),
      native('numel', (args) => t(args[0], 'torch.numel').numel),
      creation('randn'), creation('rand'), creation('zeros'), creation('ones'), creation('empty'),
      native('full', (args, kwargs) => this.create('full', toInts(args[0], 'size'), dtypeOf(kwargs, 'float'), line())),
      native('eye', (args) => this.create('eye', [toInt(args[0], 'n'), toInt(args[1] ?? args[0], 'm')], 'float', line())),
      native('randint', (args, kwargs) => {
        const size = args.length >= 3 ? args[2] : args[1] ?? kwargs.get('size')
        return this.create('randint', toInts(size ?? new PyTuple([]), 'size'), dtypeOf(kwargs, 'long'), line())
      }),
      native('arange', (args, kwargs) => {
        const numbers = args.map((arg) => toNumber(arg, 'arange'))
        const [start, end, step] = numbers.length === 1 ? [0, numbers[0], 1] : [numbers[0], numbers[1], numbers[2] ?? 1]
        return this.create('arange', [Math.max(0, Math.ceil((end - start) / step))], dtypeOf(kwargs, Number.isInteger(start) && Number.isInteger(step) ? 'long' : 'float'), line())
      }),
      native('linspace', (args, kwargs) => this.create('linspace', [toInt(args[2] ?? kwargs.get('steps')!, 'steps')], 'float', line())),
      native('tensor', (args, kwargs) => {
        const data = args[0]
        const flat = data instanceof PyList || data instanceof PyTuple ? data.items.flat() : [data]
        const dtype = dtypeOf(kwargs, flat.every((item) => typeof item === 'number' && Number.isInteger(item)) ? 'long' : 'float')
        return this.create('tensor', nestedListShape(data), dtype, line())
      }),
      like('zeros_like'), like('ones_like'), like('empty_like'), like('full_like'), like('randn_like', 'float'), like('rand_like', 'float'),
      native('tril', (args) => { const x = t(args[0], 'torch.tril'); return this.passThrough(x, x.shape) }),
      native('triu', (args) => { const x = t(args[0], 'torch.triu'); return this.passThrough(x, x.shape) }),
      ...['exp', 'log', 'sqrt', 'rsqrt', 'sin', 'cos', 'sigmoid', 'tanh', 'relu', 'softmax', 'log_softmax'].map((name) => activationFn(name)),
      native('abs', (args) => args[0] instanceof Tensor ? this.passThrough(args[0], args[0].shape) : Math.abs(toNumber(args[0], 'abs'))),
      native('clamp', (args) => { const x = t(args[0], 'torch.clamp'); return this.passThrough(x, x.shape) }),
      native('pow', (args) => args[0] instanceof Tensor ? this.passThrough(args[0], args[0].shape) : toNumber(args[0], 'pow') ** toNumber(args[1], 'pow')),
      native('cat', (args, kwargs) => {
        const tensors = this.iterate(args[0] ?? kwargs.get('tensors')!).map((item, index) => t(item, `torch.cat[${index}]`))
        return this.recordOp('concat', 'concat', tensors, new Tensor(S.concat(tensors.map((tensor) => tensor.shape), toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim')), null, tensors[0]?.dtype), line(), { dim: toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim') })
      }),
      native('stack', (args, kwargs) => {
        const tensors = this.iterate(args[0]).map((item, index) => t(item, `torch.stack[${index}]`))
        return this.recordOp('stack', 'concat', tensors, new Tensor(S.stack(tensors.map((tensor) => tensor.shape), toInt(args[1] ?? kwargs.get('dim') ?? 0, 'dim'))), line())
      }),
      native('matmul', (args) => this.tensorBinary('@', args[0], args[1], line())), native('bmm', (args) => this.tensorBinary('@', args[0], args[1], line())), native('mm', (args) => this.tensorBinary('@', args[0], args[1], line())),
      native('add', (args) => this.binary('+', args[0], args[1], line())), native('mul', (args) => this.binary('*', args[0], args[1], line())),
      native('einsum', (args) => {
        const equation = String(args[0])
        const tensors = (args.length === 2 && (args[1] instanceof PyList || args[1] instanceof PyTuple) ? this.iterate(args[1]) : args.slice(1)).map((item) => t(item, 'einsum'))
        return this.recordOp('einsum', 'matmul', tensors, new Tensor(S.einsum(equation, tensors.map((tensor) => tensor.shape))), line(), { equation })
      }),
      native('where', (args) => this.tensorFunction('where', args, new Map())),
      native('maximum', (args) => this.tensorFunction('maximum', args, new Map())),
      native('minimum', (args) => this.tensorFunction('minimum', args, new Map())),
      passFn('flatten', 'flatten'), passFn('reshape', 'reshape'), passFn('transpose', 'transpose'), passFn('permute', 'permute'), passFn('squeeze', 'squeeze'), passFn('unsqueeze', 'unsqueeze'),
      passFn('chunk', 'chunk'), passFn('split', 'split'), passFn('mean', 'mean'), passFn('sum', 'sum'), passFn('amax', 'amax'), passFn('max', 'max'), passFn('min', 'min'), passFn('argmax', 'argmax'),
      passFn('masked_fill', 'masked_fill'), passFn('unbind', 'unbind'), passFn('logsumexp', 'logsumexp'), passFn('cumsum', 'cumsum'), passFn('flip', 'flip'), passFn('roll', 'roll'), passFn('norm', 'norm'),
      ...['topk', 'sort', 'argsort', 'kthvalue', 'median', 'mode', 'cummax', 'cummin', 'aminmax', 'gather', 'scatter', 'scatter_add', 'scatter_reduce', 'index_select', 'index_add', 'index_copy', 'index_fill', 'index_put',
        'masked_select', 'masked_scatter', 'nonzero', 'argwhere', 'any', 'all', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 'greater', 'less', 'equal', 'logical_and', 'logical_or', 'logical_xor', 'logical_not', 'isclose',
        'isnan', 'isinf', 'isfinite', 'signbit', 'sign', 'sgn', 'trunc', 'frac', 'log1p', 'expm1', 'erf', 'erfinv', 'log2', 'log10', 'nan_to_num', 'clamp_min', 'clamp_max', 'lerp', 'cumprod', 'logcumsumexp',
        'addcmul', 'addcdiv', 'fmod', 'remainder', 'narrow', 'select', 'repeat_interleave', 'tile', 'diagonal', 'diff', 'outer', 'ger', 'dot', 'vdot', 'inner', 'cross', 'multinomial', 'ravel', 'movedim', 'moveaxis',
        'broadcast_to', 'std_mean', 'var_mean', 'view_as_complex', 'view_as_real', 'swapaxes', 'swapdims', 'argmin', 'amin', 'std', 'var', 'prod', 'square', 'neg', 'reciprocal', 'floor', 'ceil', 'round',
        'exp2', 'squeeze_copy', 'clip', 't', 'expand_copy', 'take_along_dim', 'bernoulli', 'unflatten', 'split_with_sizes', 'tensor_split', 'hsplit', 'vsplit', 'matrix_power', 'det', 'logdet', 'trace']
        .filter((name) => !['squeeze_copy', 'expand_copy', 'take_along_dim', 'tensor_split', 'hsplit', 'vsplit', 'matrix_power', 'det', 'logdet', 'trace', 'exp2', 'swapdims'].includes(name))
        .map((name) => passFn(name, name)),
      passFn('swapdims', 'transpose'), passFn('exp2', 'exp'),
      native('take_along_dim', (args, kwargs) => this.call(this.tensorAttr(t(args[0], 'torch.take_along_dim'), 'gather'), [args[2] ?? kwargs.get('dim') ?? 0, args[1]], new Map(), this.location)),
      native('diag', (args) => { const x = t(args[0], 'torch.diag'); return this.call(this.tensorAttr(x, 'diag'), args.slice(1), new Map(), this.location) }),
      native('meshgrid', (args, kwargs) => {
        const tensors = (args.length === 1 && (args[0] instanceof PyList || args[0] instanceof PyTuple) ? this.iterate(args[0]) : args).map((item) => t(item, 'torch.meshgrid'))
        const sizes = tensors.map((tensor) => tensor.numel)
        const shape = kwargs.get('indexing') === 'xy' && sizes.length >= 2 ? [sizes[1], sizes[0], ...sizes.slice(2)] : sizes
        return new PyTuple(tensors.map((tensor) => this.recordOp('meshgrid', 'tensor', [tensor], new Tensor(shape, null, tensor.dtype), line())))
      }),
      native('cartesian_prod', (args) => { const tensors = args.map((item) => t(item, 'torch.cartesian_prod')); return this.recordOp('cartesian_prod', 'tensor', tensors, new Tensor([tensors.reduce((p, x) => p * x.numel, 1), tensors.length]), line()) }),
      native('cdist', (args) => {
        const [a, c] = [t(args[0], 'torch.cdist'), t(args[1], 'torch.cdist')]
        return this.recordOp('cdist', 'matmul', [a, c], new Tensor([...a.shape.slice(0, -1), c.shape[c.shape.length - 2]]), line())
      }),
      native('polar', (args) => { const x = t(args[0], 'torch.polar'); return this.passThrough(x, x.shape, 'float', 'polar') }),
      native('complex', (args) => { const x = t(args[0], 'torch.complex'); return this.passThrough(x, x.shape, 'float', 'complex') }),
      native('randperm', (args, kwargs) => this.create('randperm', [toInt(args[0], 'n')], dtypeOf(kwargs, 'long'), line())),
      native('normal', (args, kwargs) => {
        const source = [args[0], args[1]].find((item) => item instanceof Tensor) as Tensor | undefined
        const size = args[2] ?? kwargs.get('size')
        return this.create('normal', source ? source.shape : toInts(size ?? new PyTuple([]), 'size'), 'float', line())
      }),
      native('logspace', (args, kwargs) => this.create('logspace', [toInt(args[2] ?? kwargs.get('steps')!, 'steps')], 'float', line())),
      native('tril_indices', (args) => { const [rows, cols] = [toInt(args[0], 'row'), toInt(args[1], 'col')]; const offset = toInt(args[2] ?? 0, 'offset'); let count = 0; for (let r = 0; r < rows; r++) count += Math.max(0, Math.min(cols, r + offset + 1)); return this.create('tril_indices', [2, count], 'long', line()) }),
      native('triu_indices', (args) => { const [rows, cols] = [toInt(args[0], 'row'), toInt(args[1], 'col')]; const offset = toInt(args[2] ?? 0, 'offset'); let count = 0; for (let r = 0; r < rows; r++) count += Math.max(0, cols - Math.max(0, r + offset)); return this.create('triu_indices', [2, count], 'long', line()) }),
      native('searchsorted', (args) => { const values = t(args[1], 'torch.searchsorted'); return this.passThrough(values, values.shape, 'long', 'searchsorted') }),
      native('bucketize', (args) => { const values = t(args[0], 'torch.bucketize'); return this.passThrough(values, values.shape, 'long', 'bucketize') }),
      native('unique', (args, kwargs) => {
        const x = t(args[0], 'torch.unique')
        const values = this.passThrough(x, kwargs.get('dim') == null ? [x.numel] : x.shape, x.dtype, 'unique')
        const extras = [kwargs.get('return_inverse') ?? false, kwargs.get('return_counts') ?? false].filter(isTruthy)
        return extras.length ? new PyTuple([values, ...extras.map(() => new Tensor([x.numel], null, 'long'))]) : values
      }),
      native('broadcast_tensors', (args) => {
        const tensors = args.map((item) => t(item, 'torch.broadcast_tensors'))
        const shape = tensors.reduce((acc, x) => S.broadcast(acc, x.shape, 'broadcast_tensors'), [] as number[])
        return new PyTuple(tensors.map((x) => this.passThrough(x, shape)))
      }),
      native('broadcast_shapes', (args) => new TorchSize(args.reduce((acc: number[], item) => S.broadcast(acc, toInts(item, 'shape'), 'broadcast_shapes'), [] as number[]))),
      native('cosine_similarity', (args, kwargs) => this.call(F.attrs.get('cosine_similarity')!, args, kwargs, this.location)),
      native('dist', (args) => this.recordOp('dist', 'generic', [t(args[0], 'torch.dist'), t(args[1], 'torch.dist')], new Tensor([]), line())),
      native('as_tensor', (args, kwargs) => args[0] instanceof Tensor ? args[0] : this.create('as_tensor', nestedListShape(args[0]), dtypeOf(kwargs, 'float'), line())),
      native('from_numpy', (args) => args[0] instanceof Tensor ? args[0] : this.create('from_numpy', nestedListShape(args[0]), 'float', line())),
      native('addmm', (args) => this.tensorBinary('@', args[1], args[2], line())), native('baddbmm', (args) => this.tensorBinary('@', args[1], args[2], line())),
      native('mv', (args) => this.tensorBinary('@', args[0], args[1], line())), native('sub', (args) => this.binary('-', args[0], args[1], line())), native('div', (args) => this.binary('/', args[0], args[1], line())),
      native('square', (args) => { const x = t(args[0], 'torch.square'); return this.passThrough(x, x.shape) }),
      ['pi', Math.PI], ['inf', Infinity],
    ])

    const math = ns('math', [
      ['pi', Math.PI], ['e', Math.E], ['inf', Infinity],
      ...(['sqrt', 'log', 'exp', 'ceil', 'floor', 'sin', 'cos', 'tanh', 'log2', 'log10', 'fabs'] as const).map((name) => native(name, (args) => {
        const value = toNumber(args[0], `math.${name}`)
        const fns: Record<string, (v: number) => number> = { sqrt: Math.sqrt, log: Math.log, exp: Math.exp, ceil: Math.ceil, floor: Math.floor, sin: Math.sin, cos: Math.cos, tanh: Math.tanh, log2: Math.log2, log10: Math.log10, fabs: Math.abs }
        return fns[name](value)
      })),
      native('prod', (args) => this.iterate(args[0]).reduce((total: number, item) => total * toNumber(item, 'prod'), 1)),
      native('gcd', (args) => { let [a, c] = [toInt(args[0], 'a'), toInt(args[1], 'b')]; while (c) [a, c] = [c, a % c]; return Math.abs(a) }),
      native('isqrt', (args) => Math.floor(Math.sqrt(toNumber(args[0], 'isqrt')))),
    ])

    const einopsFn = (name: 'rearrange' | 'repeat' | 'reduce'): [string, Value] => native(name, (args, kwargs) => {
      const x = t(args[0], `einops.${name}`)
      const pattern = String(args[1])
      const sizes: Record<string, number> = {}
      kwargs.forEach((value, key) => { if (typeof value === 'number') sizes[key] = value })
      const shape = S.einopsRearrange(x.shape, pattern, sizes)
      if (name === 'reduce') return this.recordOp('reduce', 'pool', [x], new Tensor(shape), line(), { pattern })
      return this.passThrough(x, shape, x.dtype, name)
    })
    const einops = ns('einops', [einopsFn('rearrange'), einopsFn('repeat'), einopsFn('reduce')])
    const dataclasses = ns('dataclasses', [
      native('dataclass', (args) => args[0] ?? new NativeFn('dataclass', (inner) => inner[0] ?? null)),
      native('field', (_, kwargs) => new PyDict([['__field__', true], ['default', kwargs.get('default') ?? null], ['default_factory', kwargs.get('default_factory') ?? null]])),
    ])
    const collections = ns('collections', [native('OrderedDict', (args) => {
      const dict = new PyDict()
      if (args[0]) this.iterate(args[0]).forEach((pair) => { const [key, value] = this.iterate(pair); dict.map.set(this.key(key), value) })
      return dict
    })])
    const typing = ns('typing', [], true)
    const functools = ns('functools', [native('partial', (args, kwargs, site) => {
      const [target, ...bound] = args
      return new NativeFn('partial', (more, extra, inner) => this.call(target, [...bound, ...more], new Map([...kwargs, ...extra]), inner ?? site))
    })])

    b.set('__module_torch', torch)
    b.set('__module_math', math)
    b.set('__module_einops', einops)
    b.set('__module_dataclasses', dataclasses)
    b.set('__module_collections', collections)
    b.set('__module_typing', typing)
    b.set('__module_functools', functools)
    b.set('__module_copy', ns('copy', [native('deepcopy', (args) => this.deepCopy(args[0] ?? null, new Map())), native('copy', (args) => this.shallowCopy(args[0] ?? null))]))
    b.set('__module_numpy', ns('numpy', [['pi', Math.PI]], true))
    b.set('__module_inspect', ns('inspect', [], true))
    // Helpers that training scripts commonly import; they do nothing here.
    b.set('__module_tqdm', ns('tqdm', [native('tqdm', (args) => args[0] ?? null), native('trange', (args) => { const [a, b, c] = args.map((arg) => toInt(arg, 'trange')); return args.length === 1 ? new PyRange(0, a, 1) : new PyRange(a, b, c ?? 1) })], true))
    b.set('__module_time', ns('time', [native('time', () => 0), native('perf_counter', () => 0), native('sleep', () => null)], true))
    b.set('__module_random', ns('random', [native('random', () => 0.5), native('seed', () => null), native('randint', (args) => args[0] ?? 0), native('choice', (args) => this.iterate(args[0])[0] ?? null), native('shuffle', () => null)], true))
    for (const name of ['os', 'sys', 'logging', 'json', 'argparse', 'pathlib', 'wandb', 'warnings']) b.set(`__module_${name}`, ns(name, [], true))
    // Common aliases are available even when the import line is missing.
    b.set('torch', torch)
    b.set('nn', nn)
    b.set('F', F)
    b.set('math', math)
    b.set('dataclass', dataclasses.attrs.get('dataclass')!)
    b.set('field', dataclasses.attrs.get('field')!)
  }
}

let preludeBody: Stmt[] | undefined

export function groupType(className: string): ModelNodeType {
  const name = className.toLowerCase()
  if (/attn|attention/.test(name)) return 'attention'
  if (/mlp|feedforward|ffn|swiglu/.test(name)) return 'mlp'
  if (/encoder/.test(name)) return 'encoder'
  if (/decoder/.test(name)) return 'decoder'
  if (/embed/.test(name)) return 'embedding'
  if (/down/.test(name)) return 'downsample'
  if (/up/.test(name)) return 'upsample'
  if (/norm/.test(name)) return 'norm'
  return 'module'
}

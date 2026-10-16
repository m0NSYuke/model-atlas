// Runtime values for the Atlas Python interpreter.
import type { Param, Stmt } from './parser'

export class PyError extends Error {
  constructor(message: string, public line?: number, public file?: string) {
    super(message)
  }
}

export interface Site { file: string; line: number; endLine: number }

export class Tensor {
  constructor(
    public shape: number[],
    public producer: string | null = null,
    public dtype: 'float' | 'long' | 'bool' = 'float',
    public role: 'value' | 'parameter' | 'buffer' = 'value',
  ) {}
  /** Attribute name for parameters and buffers; the owner's path is resolved when traced. */
  label?: string
  /** Shape-only operations applied since the producer (view, transpose, flatten…). */
  reshapes?: string[]
  owner?: ModuleObj
  get qualifiedName(): string | undefined {
    if (!this.label) return undefined
    const path = this.owner?.path
    return path ? `${path}.${this.label}` : this.label
  }
  get numel(): number { return this.shape.reduce((total, size) => total * size, 1) }
}

export class PyList { constructor(public items: Value[]) {} }
export class PyTuple { constructor(public items: Value[]) {} }
/** A set behaves like a list of unique items here; only printing differs. */
export class PySet extends PyList {
  constructor(items: Value[]) { super([...new Set(items)]) }
}
export class PyDict {
  map = new Map<Primitive, Value>()
  constructor(entries: [Primitive, Value][] = []) { entries.forEach(([key, value]) => this.map.set(key, value)) }
}
export class PyRange {
  constructor(public start: number, public stop: number, public step: number) {}
  get items(): number[] {
    const values: number[] = []
    if (this.step > 0) for (let value = this.start; value < this.stop; value += this.step) values.push(value)
    else for (let value = this.start; value > this.stop; value += this.step) values.push(value)
    return values
  }
}
export class PySlice { constructor(public start: number | null, public stop: number | null, public step: number | null) {} }
export const Ellipsis = { kind: 'ellipsis' as const }

export class Scope {
  vars = new Map<string, Value>()
  constructor(public parent?: Scope) {}
  lookup(name: string): Value | undefined {
    for (let scope: Scope | undefined = this; scope; scope = scope.parent) {
      if (scope.vars.has(name)) return scope.vars.get(name)
    }
    return undefined
  }
}

export class PyFunction {
  isStatic = false
  isClassMethod = false
  isProperty = false
  constructor(
    public name: string,
    public params: Param[],
    public body: Stmt[] | null,
    public closure: Scope,
    public defaults: Map<string, Value>,
    public site: Site,
    public docstring?: string,
    public expressionBody?: import('./parser').Expr,
  ) {}
}

export class PyClass {
  attrs = new Map<string, Value>()
  isDataclass = false
  fields: { name: string; defaultValue?: Value; factory?: Value }[] = []
  constructor(public name: string, public bases: Value[], public site: Site, public docstring?: string) {}

  /** Method resolution: depth-first through user base classes. */
  find(name: string): Value | undefined {
    if (this.attrs.has(name)) return this.attrs.get(name)
    for (const base of this.bases) {
      if (base instanceof PyClass) {
        const found = base.find(name)
        if (found !== undefined) return found
      }
    }
    return undefined
  }

  get isModule(): boolean {
    return this.bases.some((base) => base === NN_MODULE || (base instanceof PyClass && base.isModule))
  }
}

export class PyInstance {
  attrs = new Map<string, Value>()
  constructor(public cls: PyClass) {}
}

export class BoundMethod { constructor(public self: Value, public fn: PyFunction, public owner: PyClass) {} }

export type NativeImpl = (args: Value[], kwargs: Map<string, Value>, site: Site) => Value

export class NativeFn {
  constructor(public name: string, public impl: NativeImpl) {}
}

export class Namespace {
  constructor(public name: string, public attrs: Map<string, Value>, public permissive = false) {}
}

/** Marker for nn.Module so user classes can inherit from it. */
export const NN_MODULE = { kind: 'nn.Module' as const }

export class SuperProxy { constructor(public self: Value, public cls: PyClass) {} }
export class ContextManager { constructor(public name: string) {} }
export class Dtype { constructor(public name: string, public kind: 'float' | 'long' | 'bool') {} }

export interface LayerConfig {
  className: string
  /** Display configuration, shown in the Inspector. */
  config: Record<string, string | number | boolean | number[]>
  /** Named learnable tensors (weight, bias …). */
  params: Map<string, Tensor>
  /** Python-visible attributes such as in_features or out_channels. */
  props: Map<string, Value>
  forward: (inputs: Value[], kwargs: Map<string, Value>) => Value
}

export class ModuleObj {
  attrs = new Map<string, Value>()
  children = new Map<string, ModuleObj>()
  parent?: ModuleObj
  nameInParent?: string
  callCount = 0
  constructor(
    public kind: 'user' | 'layer' | 'sequential' | 'modulelist' | 'moduledict',
    public className: string,
    public site: Site,
    public cls?: PyClass,
    public layer?: LayerConfig,
  ) {}

  get path(): string {
    if (!this.parent) return ''
    const parentPath = this.parent.path
    return parentPath ? `${parentPath}.${this.nameInParent}` : this.nameInParent ?? ''
  }

  addChild(name: string, child: ModuleObj): void {
    this.children.set(name, child)
    if (!child.parent && child !== this) { child.parent = this; child.nameInParent = name }
  }

  /** Learnable tensors of this module and its submodules, deduplicated. */
  parameters(seen = new Set<Tensor>()): Tensor[] {
    const result: Tensor[] = []
    const visit = (tensor: Tensor) => { if (!seen.has(tensor)) { seen.add(tensor); result.push(tensor) } }
    this.layer?.params.forEach(visit)
    this.attrs.forEach((value) => { if (value instanceof Tensor && value.role === 'parameter') visit(value) })
    this.children.forEach((child) => result.push(...child.parameters(seen)))
    return result
  }
}

export type Primitive = number | string | boolean | null
export type Value =
  | Primitive | PyList | PyTuple | PyDict | PyRange | PySlice | typeof Ellipsis
  | Tensor | PyFunction | BoundMethod | PyClass | PyInstance | NativeFn | Namespace
  | typeof NN_MODULE | ModuleObj | SuperProxy | ContextManager | Dtype

export function typeName(value: Value): string {
  if (value === null) return 'None'
  if (typeof value === 'number') return Number.isInteger(value) ? 'int' : 'float'
  if (typeof value === 'string') return 'str'
  if (typeof value === 'boolean') return 'bool'
  if (value instanceof Tensor) return 'Tensor'
  if (value instanceof PySet) return 'set'
  if (value instanceof PyList) return 'list'
  if (value instanceof PyTuple) return 'tuple'
  if (value instanceof PyDict) return 'dict'
  if (value instanceof ModuleObj) return value.className
  if (value instanceof PyInstance) return value.cls.name
  if (value instanceof PyClass) return 'type'
  if (value instanceof PyFunction || value instanceof NativeFn || value instanceof BoundMethod) return 'function'
  return 'object'
}

export function formatValue(value: Value): string {
  if (value === null) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number' || typeof value === 'string') return String(value)
  if (value instanceof PyTuple) return `(${value.items.map(formatValue).join(', ')}${value.items.length === 1 ? ',' : ''})`
  if (value instanceof PyList) return `[${value.items.map(formatValue).join(', ')}]`
  if (value instanceof Tensor) return `Tensor[${value.shape.join(', ')}]`
  return typeName(value)
}

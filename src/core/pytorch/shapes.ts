// Pure tensor-shape arithmetic used by the interpreter and layer rules.
import { PyError } from './values'

export const fmt = (shape: number[]) => `[${shape.join(', ')}]`

export function normalizeDim(dim: number, rank: number, what = 'dim'): number {
  const result = dim < 0 ? dim + rank : dim
  if (!Number.isInteger(result) || result < 0 || result >= Math.max(rank, 1)) throw new PyError(`${what}=${dim} is out of range for a tensor with ${rank} dimensions`)
  return result
}

export function broadcast(a: number[], b: number[], op = 'operation'): number[] {
  const rank = Math.max(a.length, b.length)
  const result: number[] = []
  for (let index = 0; index < rank; index++) {
    const x = a[a.length - rank + index] ?? 1
    const y = b[b.length - rank + index] ?? 1
    if (x !== y && x !== 1 && y !== 1) throw new PyError(`Shapes ${fmt(a)} and ${fmt(b)} cannot be broadcast for ${op}`)
    result.push(x === 1 ? y : x)
  }
  return result
}

export function matmul(a: number[], b: number[]): number[] {
  if (!a.length || !b.length) throw new PyError('matmul needs tensors with at least one dimension')
  const left = a.length === 1 ? [1, a[0]] : a
  const right = b.length === 1 ? [b[0], 1] : b
  const k1 = left[left.length - 1]
  const k2 = right[right.length - 2]
  if (k1 !== k2) throw new PyError(`matmul shape mismatch: ${fmt(a)} @ ${fmt(b)} (inner dimensions ${k1} ≠ ${k2})`)
  const batch = broadcast(left.slice(0, -2), right.slice(0, -2), 'matmul batch dimensions')
  const result = [...batch, left[left.length - 2], right[right.length - 1]]
  if (a.length === 1) result.splice(result.length - 2, 1)
  if (b.length === 1) result.splice(result.length - 1, 1)
  return result
}

export function reshape(shape: number[], target: number[]): number[] {
  const total = shape.reduce((x, y) => x * y, 1)
  const unknown = target.filter((size) => size === -1).length
  if (unknown > 1) throw new PyError('view/reshape can infer only one dimension (-1)')
  const known = target.filter((size) => size !== -1).reduce((x, y) => x * y, 1)
  const result = target.map((size) => size === -1 ? total / Math.max(known, 1) : size)
  if (result.some((size) => !Number.isInteger(size) || size < 0) || result.reduce((x, y) => x * y, 1) !== total) {
    throw new PyError(`Cannot reshape a tensor of shape ${fmt(shape)} (${total} elements) into ${fmt(target)}`)
  }
  return result
}

export function flatten(shape: number[], start = 0, end = -1): number[] {
  if (!shape.length) return [1]
  const s = normalizeDim(start, shape.length, 'start_dim')
  const e = normalizeDim(end, shape.length, 'end_dim')
  if (e < s) return shape
  return [...shape.slice(0, s), shape.slice(s, e + 1).reduce((x, y) => x * y, 1), ...shape.slice(e + 1)]
}

export function transpose(shape: number[], a: number, b: number): number[] {
  const result = [...shape]
  const x = normalizeDim(a, shape.length)
  const y = normalizeDim(b, shape.length)
  ;[result[x], result[y]] = [result[y], result[x]]
  return result
}

export function permute(shape: number[], dims: number[]): number[] {
  if (dims.length !== shape.length) throw new PyError(`permute expects ${shape.length} dimensions, got ${dims.length}`)
  const axes = dims.map((dim) => normalizeDim(dim, shape.length))
  if (new Set(axes).size !== axes.length) throw new PyError(`permute: repeated dims are not allowed, got (${dims.join(', ')})`)
  return axes.map((axis) => shape[axis])
}

export function reduce(shape: number[], dim: number | number[] | null, keepdim: boolean): number[] {
  if (dim === null) return keepdim ? shape.map(() => 1) : []
  const dims = (Array.isArray(dim) ? dim : [dim]).map((item) => normalizeDim(item, shape.length))
  return keepdim ? shape.map((size, index) => dims.includes(index) ? 1 : size) : shape.filter((_, index) => !dims.includes(index))
}

export function concat(shapes: number[][], dim: number): number[] {
  if (!shapes.length) throw new PyError('torch.cat needs at least one tensor')
  const rank = shapes[0].length
  const axis = normalizeDim(dim, rank)
  shapes.forEach((shape) => {
    if (shape.length !== rank || shape.some((size, index) => index !== axis && size !== shapes[0][index])) {
      throw new PyError(`torch.cat shape mismatch: ${shapes.map(fmt).join(' + ')} along dim ${dim}`)
    }
  })
  return shapes[0].map((size, index) => index === axis ? shapes.reduce((total, shape) => total + shape[axis], 0) : size)
}

export function stack(shapes: number[][], dim: number): number[] {
  if (!shapes.length) throw new PyError('torch.stack needs at least one tensor')
  shapes.forEach((shape) => { if (fmt(shape) !== fmt(shapes[0])) throw new PyError(`torch.stack needs equal shapes, got ${shapes.map(fmt).join(', ')}`) })
  // The new axis may go anywhere in [-(ndim + 1), ndim].
  const axis = normalizeDim(dim, shapes[0].length + 1)
  const result = [...shapes[0]]
  result.splice(axis, 0, shapes.length)
  return result
}

/** Output length of a convolution or pooling window along one axis. */
export function convAxis(size: number, kernel: number, stride: number, padding: number, dilation = 1, name = 'kernel_size'): number {
  const result = Math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
  if (result < 1) throw new PyError(`${name}=${kernel} with padding=${padding} is larger than the input size ${size}`)
  return result
}

/**
 * Output length of a pooling window along one axis. With ceil_mode the last partial window counts too,
 * unless it would start beyond the input and its left padding (PyTorch's pooling_output_shape).
 */
export function poolAxis(size: number, kernel: number, stride: number, padding: number, dilation = 1, ceilMode = false): number {
  const span = size + 2 * padding - dilation * (kernel - 1) - 1
  let result = (ceilMode ? Math.ceil(span / stride) : Math.floor(span / stride)) + 1
  if (ceilMode && (result - 1) * stride >= size + padding) result--
  if (result < 1) throw new PyError(`kernel_size=${kernel} with padding=${padding} is larger than the input size ${size}`)
  return result
}

export function transposedConvAxis(size: number, kernel: number, stride: number, padding: number, outputPadding = 0, dilation = 1): number {
  return (size - 1) * stride - 2 * padding + dilation * (kernel - 1) + outputPadding + 1
}

export type IndexItem = number | 'slice' | 'none' | 'ellipsis' | { start: number | null; stop: number | null; step: number | null } | { array: number[]; consumes: number }

/**
 * Result shape of x[index], following NumPy/PyTorch rules. Integer arrays (lists, long tensors) broadcast together;
 * a boolean mask consumes as many axes as it has and yields one axis with every element selected (an upper bound,
 * since the real count depends on the data).
 */
export function indexShape(shape: number[], index: IndexItem[]): number[] {
  const consumes = (item: IndexItem) => item === 'none' || item === 'ellipsis' ? 0 : typeof item === 'object' && 'array' in item ? item.consumes : 1
  const consumed = index.reduce((total: number, item) => total + consumes(item), 0)
  const ellipsisAt = index.indexOf('ellipsis')
  const expanded: IndexItem[] = ellipsisAt >= 0 ? [...index.slice(0, ellipsisAt), ...Array(Math.max(0, shape.length - consumed)).fill('slice'), ...index.slice(ellipsisAt + 1)] : index
  if (expanded.reduce((total: number, item) => total + consumes(item), 0) > shape.length) throw new PyError(`Too many indices for a tensor of shape ${fmt(shape)}`)
  const result: number[] = []
  let axis = 0
  let arrayShape: number[] | null = null
  let arrayAt = -1
  let arraysSplit = false
  let lastWasArray = false
  for (const item of expanded) {
    if (typeof item === 'object' && 'array' in item) {
      if (arrayShape === null) { arrayShape = item.array; arrayAt = result.length }
      else { arrayShape = broadcast(arrayShape, item.array, 'indexing'); if (!lastWasArray) arraysSplit = true }
      axis += item.consumes
      lastWasArray = true
      continue
    }
    lastWasArray = false
    if (item === 'none') { result.push(1); continue }
    const size = shape[axis++]
    if (typeof item === 'number') {
      if (item >= size || item < -size) throw new PyError(`Index ${item} is out of range for dimension of size ${size}`)
      // An integer next to index arrays counts as one of them (it broadcasts as a 0-d array).
      if (arrayShape !== null) lastWasArray = true
      continue
    }
    if (item === 'slice' || item === 'ellipsis') { result.push(size); continue }
    const step = item.step ?? 1
    const clamp = (value: number | null, fallback: number) => {
      if (value === null) return fallback
      const resolved = value < 0 ? value + size : value
      return Math.max(0, Math.min(size, resolved))
    }
    const start = clamp(item.start, 0)
    const stop = clamp(item.stop, size)
    result.push(Math.max(0, Math.ceil((stop - start) / step)))
  }
  const rest = shape.slice(axis)
  if (arrayShape === null) return [...result, ...rest]
  if (arraysSplit) return [...arrayShape, ...result, ...rest]
  return [...result.slice(0, arrayAt), ...arrayShape, ...result.slice(arrayAt), ...rest]
}

/** Minimal einops pattern support: names, parenthesized groups and "1". */
export function einopsRearrange(shape: number[], pattern: string, sizes: Record<string, number>): number[] {
  const [left, right] = pattern.split('->').map((part) => part.trim())
  if (!right) throw new PyError(`Invalid einops pattern '${pattern}'`)
  const parse = (side: string) => {
    const groups: string[][] = []
    const tokens = side.match(/\([^)]*\)|\S+/g) ?? []
    tokens.forEach((token) => groups.push(token.startsWith('(') ? token.slice(1, -1).trim().split(/\s+/).filter(Boolean) : [token]))
    return groups
  }
  const input = parse(left)
  if (input.length !== shape.length) throw new PyError(`einops pattern '${left}' expects ${input.length} dimensions, got ${shape.length}`)
  const known: Record<string, number> = { ...sizes }
  input.forEach((group, index) => {
    const unknown = group.filter((name) => name !== '1' && known[name] === undefined)
    const product = group.filter((name) => name !== '1' && known[name] !== undefined).reduce((total, name) => total * known[name], 1)
    if (unknown.length > 1) throw new PyError(`einops cannot infer both ${unknown.join(' and ')}`)
    if (unknown.length === 1) known[unknown[0]] = shape[index] / product
    else if (product !== shape[index]) throw new PyError(`einops axis ${group.join(' ')} expects size ${product}, got ${shape[index]}`)
  })
  return parse(right).map((group) => group.reduce((total, name) => {
    if (name === '1') return total
    if (known[name] === undefined) throw new PyError(`einops axis '${name}' has no known size`)
    return total * known[name]
  }, 1))
}

export function einsum(equation: string, shapes: number[][]): number[] {
  const [inputs, output] = equation.replace(/\s/g, '').split('->')
  const terms = inputs.split(',')
  if (terms.length !== shapes.length) throw new PyError(`einsum expects ${terms.length} operands, got ${shapes.length}`)
  const sizes: Record<string, number> = {}
  let ellipsis: number[] | null = null
  terms.forEach((term, index) => {
    const shape = shapes[index]
    const [before, after] = term.includes('...') ? term.split('...') : [term, null]
    const named = before.length + (after?.length ?? 0)
    if (after === null ? named !== shape.length : named > shape.length) throw new PyError(`einsum term '${term}' does not match shape ${fmt(shape)}`)
    const letters = after === null ? before.split('').map((letter, axis) => [letter, shape[axis]] as const)
      : [...before.split('').map((letter, axis) => [letter, shape[axis]] as const), ...after.split('').map((letter, axis) => [letter, shape[shape.length - after.length + axis]] as const)]
    if (after !== null) {
      const middle = shape.slice(before.length, shape.length - after.length)
      ellipsis = ellipsis === null ? middle : broadcast(ellipsis, middle, 'einsum')
    }
    letters.forEach(([letter, size]) => {
      if (sizes[letter] !== undefined && sizes[letter] !== size && sizes[letter] !== 1 && size !== 1) throw new PyError(`einsum size mismatch for '${letter}': ${sizes[letter]} vs ${size}`)
      sizes[letter] = Math.max(sizes[letter] ?? 1, size)
    })
  })
  const dots: number[] = ellipsis ?? []
  const all = terms.join('').replace(/\.\.\./g, '')
  const result = output ?? `${ellipsis ? '...' : ''}${Object.keys(sizes).filter((letter) => all.split(letter).length === 2).sort().join('')}`
  const [head, tail] = result.includes('...') ? result.split('...') : [result, null]
  return [...head.split('').filter(Boolean).map((letter) => sizes[letter]), ...(tail === null ? [] : [...dots, ...tail.split('').filter(Boolean).map((letter) => sizes[letter])])]
}

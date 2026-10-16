// Python-style str(), repr() and string formatting (f-strings, str.format, %) for the Atlas interpreter.
import { Ellipsis, PyDict, PyList, PySet, PyRange, PySlice, PyTuple, Tensor, typeName, type Value } from './values'

/** `torch.Size` is a tuple that prints like PyTorch: `torch.Size([2, 64])`. */
export class TorchSize extends PyTuple {}

export function pyStr(value: Value): string {
  if (typeof value === 'string') return value
  return pyRepr(value)
}

export function pyRepr(value: Value): string {
  if (value === null) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'string') return quote(value)
  if (value instanceof TorchSize) return `torch.Size([${value.items.map(pyRepr).join(', ')}])`
  if (value instanceof PyTuple) return `(${value.items.map(pyRepr).join(', ')}${value.items.length === 1 ? ',' : ''})`
  if (value instanceof PySet) return value.items.length ? `{${value.items.map(pyRepr).join(', ')}}` : 'set()'
  if (value instanceof PyList) return `[${value.items.map(pyRepr).join(', ')}]`
  if (value instanceof PyDict) return `{${[...value.map.entries()].map(([key, item]) => `${pyRepr(key)}: ${pyRepr(item)}`).join(', ')}}`
  if (value instanceof PyRange) return value.step === 1 ? `range(${value.start}, ${value.stop})` : `range(${value.start}, ${value.stop}, ${value.step})`
  if (value instanceof PySlice) return `slice(${pyRepr(value.start)}, ${pyRepr(value.stop)}, ${pyRepr(value.step)})`
  if (value === Ellipsis) return 'Ellipsis'
  if (value instanceof Tensor) return `tensor(shape=[${value.shape.join(', ')}]${value.dtype === 'float' ? '' : `, dtype=torch.${value.dtype === 'long' ? 'int64' : 'bool'}`})`
  return `<${typeName(value)}>`
}

function formatNumber(value: number): string {
  if (Number.isNaN(value)) return 'nan'
  if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf'
  return String(value)
}

function quote(text: string): string {
  const body = text.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/\t/g, '\\t')
  return text.includes("'") && !text.includes('"') ? `"${body}"` : `'${body.replace(/'/g, "\\'")}'`
}

/** format(value, spec) for the common subset of the format mini-language: [[fill]align][sign][#][0][width][,|_][.precision][type]. */
export function formatSpec(value: Value, spec: string): string {
  if (!spec) return pyStr(value)
  const match = /^(?:(.)?([<>^=]))?([+\- ])?(#)?(0)?(\d+)?([,_])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$/.exec(spec)
  if (!match) throw new Error(`Invalid format specifier '${spec}'`)
  const [, fillChar, alignChar, sign = '-', , zero, widthText, grouping, precisionText, type] = match
  const precision = precisionText === undefined ? undefined : Number(precisionText)
  let body: string
  let numeric = false
  if (typeof value === 'number' || typeof value === 'boolean') {
    numeric = true
    const number = Number(value)
    const magnitude = Math.abs(number)
    switch (type) {
      case 'd': case 'n': body = Math.trunc(magnitude).toString(); break
      case 'b': body = Math.trunc(magnitude).toString(2); break
      case 'o': body = Math.trunc(magnitude).toString(8); break
      case 'x': body = Math.trunc(magnitude).toString(16); break
      case 'X': body = Math.trunc(magnitude).toString(16).toUpperCase(); break
      case 'f': case 'F': body = magnitude.toFixed(precision ?? 6); break
      case 'e': case 'E': {
        const text = magnitude.toExponential(precision ?? 6).replace(/e([+-])(\d)$/, 'e$10$2')
        body = type === 'E' ? text.toUpperCase() : text
        break
      }
      case '%': body = `${(magnitude * 100).toFixed(precision ?? 6)}%`; break
      case 'g': case 'G': {
        const text = String(Number(magnitude.toPrecision(precision || 6)))
        body = type === 'G' ? text.toUpperCase() : text
        break
      }
      case 'c': body = String.fromCharCode(number); break
      default: body = precision !== undefined ? String(Number(magnitude.toPrecision(precision || 1))) : formatNumber(magnitude)
    }
    if (grouping) {
      const [whole, fraction] = body.split('.')
      body = whole.replace(/\B(?=(\d{3})+(?!\d))/g, grouping) + (fraction !== undefined ? `.${fraction}` : '')
    }
    const prefix = number < 0 || Object.is(number, -0) ? '-' : sign === '+' ? '+' : sign === ' ' ? ' ' : ''
    body = prefix + body
  } else {
    body = pyStr(value)
    if (precision !== undefined) body = body.slice(0, precision)
  }
  const width = widthText ? Number(widthText) : 0
  if (body.length >= width) return body
  const fill = fillChar ?? (zero && numeric ? '0' : ' ')
  const align = alignChar ?? (zero && numeric ? '=' : numeric ? '>' : '<')
  const pad = width - body.length
  if (align === '<') return body + fill.repeat(pad)
  if (align === '^') return fill.repeat(Math.floor(pad / 2)) + body + fill.repeat(Math.ceil(pad / 2))
  if (align === '=' && /^[+\- ]/.test(body)) return body[0] + fill.repeat(pad) + body.slice(1)
  return fill.repeat(pad) + body
}

export function convert(value: Value, conversion?: 'r' | 's' | 'a'): Value {
  if (conversion === 'r' || conversion === 'a') return pyRepr(value)
  if (conversion === 's') return pyStr(value)
  return value
}

/** str.format: `{}`, `{0}`, `{name}`, `{name.attr}`, `{name[0]}`, with `!r` and `:spec`. */
export function strFormat(template: string, args: Value[], kwargs: Map<string, Value>, lookup: (value: Value, path: string) => Value): string {
  let auto = 0
  return template.replace(/\{\{|\}\}|\{([^{}!:]*)(?:!([rsa]))?(?::((?:[^{}]|\{[^{}]*\})*))?\}/g, (whole, field: string | undefined, conversion: 'r' | 's' | 'a' | undefined, spec: string | undefined) => {
    if (whole === '{{') return '{'
    if (whole === '}}') return '}'
    const head = /^[^.[]*/.exec(field ?? '')![0]
    const rest = (field ?? '').slice(head.length)
    let value: Value = head === '' ? args[auto++] ?? null : /^\d+$/.test(head) ? args[Number(head)] ?? null : kwargs.get(head) ?? null
    if (rest) value = lookup(value, rest)
    const nested = (spec ?? '').replace(/\{([^{}]*)\}/g, (_, name: string) => pyStr(name === '' ? args[auto++] ?? null : /^\d+$/.test(name) ? args[Number(name)] ?? null : kwargs.get(name) ?? null))
    return formatSpec(convert(value, conversion), nested)
  })
}

/** printf-style `template % values`. */
export function percentFormat(template: string, values: Value): string {
  const items = values instanceof PyTuple ? values.items : [values]
  const named = values instanceof PyDict ? values : null
  let index = 0
  return template.replace(/%(?:\(([^)]+)\))?([#0\- +]*)(\d+|\*)?(?:\.(\d+))?([diouxXeEfFgGcrsa%])/g, (_, key: string | undefined, flags: string, widthText: string | undefined, precision: string | undefined, type: string) => {
    if (type === '%') return '%'
    const value: Value = key && named ? named.map.get(key) ?? null : items[index++] ?? null
    const width = widthText === '*' ? Number(items[index - 1]) : widthText
    if (type === 's') return formatSpec(pyStr(value), `${flags.includes('-') ? '<' : '>'}${width ?? ''}${precision !== undefined ? `.${precision}` : ''}`)
    if (type === 'r' || type === 'a') return formatSpec(pyRepr(value), `${flags.includes('-') ? '<' : '>'}${width ?? ''}`)
    const align = flags.includes('-') ? '<' : flags.includes('0') ? '0' : ''
    const sign = flags.includes('+') ? '+' : flags.includes(' ') ? ' ' : ''
    const kind = type === 'i' || type === 'u' ? 'd' : type
    return formatSpec(value, `${align === '<' ? '<' : ''}${sign}${align === '0' ? '0' : ''}${width ?? ''}${precision !== undefined ? `.${precision}` : ''}${kind}`)
  })
}

/** Python ordering for sorted()/min()/max(): numbers, strings, booleans and sequences compared item by item. */
export function pyCompare(a: Value, b: Value): number {
  if ((typeof a === 'number' || typeof a === 'boolean') && (typeof b === 'number' || typeof b === 'boolean')) return Number(a) - Number(b)
  if (typeof a === 'string' && typeof b === 'string') return a < b ? -1 : a > b ? 1 : 0
  if ((a instanceof PyTuple || a instanceof PyList) && (b instanceof PyTuple || b instanceof PyList)) {
    for (let index = 0; index < Math.min(a.items.length, b.items.length); index++) {
      const order = pyCompare(a.items[index], b.items[index])
      if (order) return order
    }
    return a.items.length - b.items.length
  }
  throw new Error(`'<' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`)
}

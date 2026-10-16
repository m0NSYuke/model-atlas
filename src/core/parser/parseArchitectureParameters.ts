export type Triple = [number, number, number]
export interface DownsampleParameters { channels: number; outChannels: number; kernelSize: Triple; stride: Triple }

function argumentList(source: string, pattern: RegExp): string | null {
  const match = pattern.exec(source)
  if (!match) return null
  const opening = source.indexOf('(', match.index + match[0].length - 1)
  let depth = 0
  for (let index = opening; index < source.length; index++) {
    if (source[index] === '(') depth++
    if (source[index] === ')' && --depth === 0) return source.slice(opening + 1, index)
  }
  return null
}

function parameter(args: string, name: string): number | Triple | undefined {
  const match = new RegExp(`\\b${name}\\s*=\\s*(\\([^)]*\\)|-?\\d+)`).exec(args)
  if (!match) return undefined
  if (match[1].startsWith('(')) {
    const values = match[1].slice(1, -1).split(',').map((item) => Number(item.trim()))
    if (values.length !== 3 || values.some((item) => !Number.isInteger(item) || item < 1 || item > 32)) throw new Error(`${name} must be a positive integer or a 3D tuple`)
    return values as Triple
  }
  const value = Number(match[1])
  if (!Number.isInteger(value) || value < 1 || value > 4096) throw new Error(`${name} must be a positive integer`)
  return value
}

function triple(value: number | Triple | undefined, fallback: number): Triple {
  if (Array.isArray(value)) return value
  const size = value ?? fallback
  return [size, size, size]
}

export function parseDownsampleParameters(codeFiles: Record<string, string>): DownsampleParameters {
  const block = codeFiles['blocks.py'] ?? ''
  const model = codeFiles['model.py'] ?? ''
  const classStart = block.indexOf('class DownsampleCausal3D')
  const defaults = classStart >= 0 ? argumentList(block.slice(classStart), /def __init__\s*\(/) ?? '' : ''
  const instance = argumentList(model, /\bDownsampleCausal3D\s*\(/) ?? ''
  if (!defaults && !instance) throw new Error('DownsampleCausal3D definition or call was not found')
  const channels = parameter(instance, 'channels') ?? parameter(defaults, 'channels') ?? 128
  const outChannels = parameter(instance, 'out_channels') ?? parameter(defaults, 'out_channels') ?? 256
  const kernelSize = parameter(instance, 'kernel_size') ?? parameter(defaults, 'kernel_size')
  const stride = parameter(instance, 'stride') ?? parameter(defaults, 'stride')
  if (Array.isArray(channels) || Array.isArray(outChannels)) throw new Error('Channels must be integers')
  return { channels, outChannels, kernelSize: triple(kernelSize, 3), stride: triple(stride, 2) }
}

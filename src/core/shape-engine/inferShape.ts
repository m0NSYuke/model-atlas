import type { ModelIR, TensorShape } from '../model-ir/types'
import { parseDownsampleParameters, type Triple } from '../parser/parseArchitectureParameters'

function axis(input: number | string, kernel: number, stride: number, paddingBefore: number, paddingAfter: number): number | string {
  if (typeof input === 'number') {
    const result = Math.floor((input + paddingBefore + paddingAfter - kernel) / stride + 1)
    if (result < 1) throw new Error('kernel_size produces a non-positive output dimension')
    return result
  }
  return stride === 1 ? input : `${input}/${stride}`
}

export function inferConv3DShape(input: TensorShape, outChannels: number, kernel: Triple, stride: Triple, padding: Triple, causal = false): TensorShape {
  return [
    input[0], outChannels,
    axis(input[2], kernel[0], stride[0], causal ? kernel[0] - 1 : padding[0], causal ? 0 : padding[0]),
    axis(input[3], kernel[1], stride[1], padding[1], padding[1]),
    axis(input[4], kernel[2], stride[2], padding[2], padding[2]),
  ]
}

export function inferDownsampleShape(input: TensorShape, outChannels: number, kernel: Triple, stride: Triple): TensorShape {
  return inferConv3DShape(input, outChannels, kernel, stride, [0, 1, 1], true)
}

export function inferPatchEmbedShape(input: TensorShape, embedDim: number, patch: Triple): TensorShape {
  return inferConv3DShape(input, embedDim, patch, patch, [0, 0, 0])
}

export function updateVaeShapes(base: ModelIR, codeFiles: Record<string, string>): ModelIR {
  const parameters = parseDownsampleParameters(codeFiles)
  const input = base.nodes.find((node) => node.id === 'downsample')?.inputShape
  if (!input) return base
  const down = inferDownsampleShape(input, parameters.outChannels, parameters.kernelSize, parameters.stride)
  const moments = [down[0], parameters.outChannels * 2, ...down.slice(2)] as TensorShape
  const output = [down[0], 3, ...down.slice(2).map((dimension, index) => typeof dimension === 'number' ? dimension * (index === 0 ? 2 : 4) : `${dimension}*${index === 0 ? 2 : 4}`)] as TensorShape
  return {
    ...base,
    nodes: base.nodes.map((node) => {
      switch (node.id) {
        case 'downsample': return { ...node, parameters: { ...node.parameters, channels: parameters.channels, out_channels: parameters.outChannels, kernel_size: parameters.kernelSize.every((size) => size === parameters.kernelSize[0]) ? parameters.kernelSize[0] : parameters.kernelSize, stride: parameters.stride.every((size) => size === parameters.stride[0]) ? parameters.stride[0] : parameters.stride }, outputShape: down }
        case 'encoder': case 'mid': return { ...node, outputShape: down, inputShape: down }
        case 'moments': return { ...node, inputShape: down, outputShape: moments }
        case 'mu': case 'logvar': return { ...node, outputShape: down }
        case 'sampling': case 'latent': return { ...node, inputShape: down, outputShape: down }
        case 'decoder': return { ...node, inputShape: down, outputShape: output }
        case 'output': return { ...node, inputShape: output, outputShape: output }
        default: return node
      }
    }),
  }
}

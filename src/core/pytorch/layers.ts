// Shape and parameter rules for torch.nn layers supported by the Atlas tracer.
import type { ModelNodeType } from '../model-ir/types'
import { broadcast, convAxis, flatten, fmt, normalizeDim, transposedConvAxis } from './shapes'
import { PyDict, PyError, PyList, PyTuple, Tensor, formatValue, type LayerConfig, type Value } from './values'

const REQUIRED = Symbol('required')
type Signature = [string, Value | typeof REQUIRED][]

export function toNumber(value: Value, name: string): number {
  if (typeof value === 'number') return value
  if (typeof value === 'boolean') return value ? 1 : 0
  throw new PyError(`${name} must be a number, got ${formatValue(value)}`)
}

export function toInt(value: Value, name: string): number {
  const number = toNumber(value, name)
  if (!Number.isInteger(number)) throw new PyError(`${name} must be an integer, got ${number}`)
  return number
}

export function toInts(value: Value, name: string): number[] {
  if (value instanceof PyTuple || value instanceof PyList) return value.items.map((item, index) => toInt(item, `${name}[${index}]`))
  return [toInt(value, name)]
}

/** Expand an int or tuple argument to n spatial dimensions. */
export function nTuple(value: Value, n: number, name: string): number[] {
  const values = toInts(value, name)
  if (values.length === 1) return Array(n).fill(values[0])
  if (values.length !== n) throw new PyError(`${name} needs ${n} values, got ${values.length}`)
  return values
}

function bind(className: string, signature: Signature, args: Value[], kwargs: Map<string, Value>): Record<string, Value> {
  const bound: Record<string, Value> = {}
  if (args.length > signature.length) throw new PyError(`nn.${className} takes at most ${signature.length} positional arguments`)
  signature.forEach(([name, fallback], index) => {
    if (index < args.length) bound[name] = args[index]
    else if (kwargs.has(name)) bound[name] = kwargs.get(name)!
    else if (fallback === REQUIRED) throw new PyError(`nn.${className} is missing the required argument '${name}'`)
    else bound[name] = fallback
  })
  kwargs.forEach((_, key) => { if (!signature.some(([name]) => name === key)) throw new PyError(`nn.${className} got an unexpected argument '${key}'`) })
  return bound
}

function tensorArg(value: Value | undefined, className: string): Tensor {
  if (!(value instanceof Tensor)) throw new PyError(`nn.${className} expects a Tensor input, got ${value === undefined ? 'nothing' : formatValue(value)}`)
  return value
}

const param = (...shape: number[]) => new Tensor(shape, null, 'float', 'parameter')

interface LayerDefinition {
  type: ModelNodeType
  signature: Signature
  build: (a: Record<string, Value>, className: string) => Omit<LayerConfig, 'className'>
}

const identityForward = (className: string) => (inputs: Value[]) => {
  const x = tensorArg(inputs[0], className)
  return new Tensor(x.shape, null, x.dtype)
}

function convDefinition(dims: number): LayerDefinition {
  return {
    type: dims === 1 ? 'conv1d' : dims === 2 ? 'conv2d' : 'conv3d',
    signature: [['in_channels', REQUIRED], ['out_channels', REQUIRED], ['kernel_size', REQUIRED], ['stride', 1], ['padding', 0], ['dilation', 1], ['groups', 1], ['bias', true], ['padding_mode', 'zeros']],
    build: (a, className) => {
      const inChannels = toInt(a.in_channels, 'in_channels')
      const outChannels = toInt(a.out_channels, 'out_channels')
      const kernel = nTuple(a.kernel_size, dims, 'kernel_size')
      const stride = nTuple(a.stride, dims, 'stride')
      const dilation = nTuple(a.dilation, dims, 'dilation')
      const groups = toInt(a.groups, 'groups')
      if (inChannels % groups || outChannels % groups) throw new PyError(`groups=${groups} must divide in_channels=${inChannels} and out_channels=${outChannels}`)
      const same = a.padding === 'same'
      const padding = typeof a.padding === 'string' ? Array(dims).fill(0) : nTuple(a.padding, dims, 'padding')
      const params = new Map([['weight', param(outChannels, inChannels / groups, ...kernel)]])
      if (a.bias) params.set('bias', param(outChannels))
      const show = (values: number[]) => values.every((value) => value === values[0]) ? values[0] : values
      return {
        config: { in_channels: inChannels, out_channels: outChannels, kernel_size: show(kernel), stride: show(stride), padding: same ? 'same' : show(padding), ...(groups > 1 ? { groups } : {}), ...(dilation.some((d) => d > 1) ? { dilation: show(dilation) } : {}) },
        params,
        props: new Map<string, Value>([['in_channels', inChannels], ['out_channels', outChannels], ['kernel_size', new PyTuple(kernel)], ['stride', new PyTuple(stride)], ['padding', new PyTuple(padding)], ['weight', params.get('weight')!], ['bias', params.get('bias') ?? null]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape.length !== dims + 2 && x.shape.length !== dims + 1) throw new PyError(`nn.${className} expects a ${dims + 2}D input [N, C, ${['L', 'H, W', 'D, H, W'][dims - 1]}], got ${fmt(x.shape)}`)
          const channelAxis = x.shape.length - dims - 1
          if (x.shape[channelAxis] !== inChannels) throw new PyError(`nn.${className}(in_channels=${inChannels}) received ${x.shape[channelAxis]} channels (input ${fmt(x.shape)})`)
          const spatial = x.shape.slice(channelAxis + 1).map((size, index) => same ? size : convAxis(size, kernel[index], stride[index], padding[index], dilation[index]))
          return new Tensor([...x.shape.slice(0, channelAxis), outChannels, ...spatial])
        },
      }
    },
  }
}

function transposedConvDefinition(dims: number): LayerDefinition {
  return {
    type: 'upsample',
    signature: [['in_channels', REQUIRED], ['out_channels', REQUIRED], ['kernel_size', REQUIRED], ['stride', 1], ['padding', 0], ['output_padding', 0], ['groups', 1], ['bias', true], ['dilation', 1], ['padding_mode', 'zeros']],
    build: (a, className) => {
      const inChannels = toInt(a.in_channels, 'in_channels')
      const outChannels = toInt(a.out_channels, 'out_channels')
      const kernel = nTuple(a.kernel_size, dims, 'kernel_size')
      const stride = nTuple(a.stride, dims, 'stride')
      const padding = nTuple(a.padding, dims, 'padding')
      const outputPadding = nTuple(a.output_padding, dims, 'output_padding')
      const dilation = nTuple(a.dilation, dims, 'dilation')
      const params = new Map([['weight', param(inChannels, outChannels, ...kernel)]])
      if (a.bias) params.set('bias', param(outChannels))
      const show = (values: number[]) => values.every((value) => value === values[0]) ? values[0] : values
      return {
        config: { in_channels: inChannels, out_channels: outChannels, kernel_size: show(kernel), stride: show(stride), padding: show(padding) },
        params,
        props: new Map<string, Value>([['in_channels', inChannels], ['out_channels', outChannels]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const channelAxis = x.shape.length - dims - 1
          if (channelAxis < 0 || x.shape[channelAxis] !== inChannels) throw new PyError(`nn.${className}(in_channels=${inChannels}) received input ${fmt(x.shape)}`)
          const spatial = x.shape.slice(channelAxis + 1).map((size, index) => transposedConvAxis(size, kernel[index], stride[index], padding[index], outputPadding[index], dilation[index]))
          return new Tensor([...x.shape.slice(0, channelAxis), outChannels, ...spatial])
        },
      }
    },
  }
}

function normDefinition(kind: 'batch' | 'instance', dims: number): LayerDefinition {
  return {
    type: 'norm',
    signature: [['num_features', REQUIRED], ['eps', 1e-5], ['momentum', 0.1], ['affine', kind === 'batch'], ['track_running_stats', kind === 'batch']],
    build: (a, className) => {
      const features = toInt(a.num_features, 'num_features')
      const params = new Map<string, Tensor>()
      if (a.affine) { params.set('weight', param(features)); params.set('bias', param(features)) }
      return {
        config: { num_features: features }, params, props: new Map<string, Value>([['num_features', features]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          // Batch norm always has a batch axis, so BatchNorm1d takes (N, C) as well as (N, C, L);
          // instance norm also accepts unbatched (C, ...) input.
          const channelAxis = kind === 'batch' || x.shape.length !== dims + 1 ? 1 : 0
          if (x.shape[channelAxis] !== features) throw new PyError(`nn.${className}(${features}) received ${x.shape[channelAxis]} channels (input ${fmt(x.shape)})`)
          return new Tensor(x.shape)
        },
      }
    },
  }
}

function poolDefinition(kind: 'max' | 'avg', dims: number): LayerDefinition {
  return {
    type: 'pool',
    signature: kind === 'max'
      ? [['kernel_size', REQUIRED], ['stride', null], ['padding', 0], ['dilation', 1], ['return_indices', false], ['ceil_mode', false]]
      : [['kernel_size', REQUIRED], ['stride', null], ['padding', 0], ['ceil_mode', false], ['count_include_pad', true], ['divisor_override', null]],
    build: (a, className) => {
      const kernel = nTuple(a.kernel_size, dims, 'kernel_size')
      const stride = a.stride === null ? kernel : nTuple(a.stride, dims, 'stride')
      const padding = nTuple(a.padding, dims, 'padding')
      return {
        config: { kernel_size: kernel[0], stride: stride[0], ...(padding[0] ? { padding: padding[0] } : {}) }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape.length < dims + 1) throw new PyError(`nn.${className} expects at least ${dims + 1} dimensions, got ${fmt(x.shape)}`)
          const lead = x.shape.slice(0, x.shape.length - dims)
          return new Tensor([...lead, ...x.shape.slice(-dims).map((size, index) => convAxis(size, kernel[index], stride[index], padding[index]))])
        },
      }
    },
  }
}

function adaptivePoolDefinition(dims: number): LayerDefinition {
  return {
    type: 'pool',
    signature: [['output_size', REQUIRED], ['return_indices', false]],
    build: (a, className) => {
      const size = a.output_size === null ? null : nTuple(a.output_size, dims, 'output_size')
      return {
        config: { output_size: size ? (size.every((item) => item === size[0]) ? size[0] : size) : 'None' }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          return new Tensor([...x.shape.slice(0, x.shape.length - dims), ...x.shape.slice(-dims).map((value, index) => size?.[index] ?? value)])
        },
      }
    },
  }
}

const activation = (type: ModelNodeType = 'activation', signature: Signature = []): LayerDefinition => ({
  type, signature,
  build: (a, className) => ({ config: Object.fromEntries(Object.entries(a).filter(([, value]) => typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean')) as Record<string, number>, params: new Map(), props: new Map(), forward: identityForward(className) }),
})

function recurrentDefinition(gates: number): LayerDefinition {
  return {
    type: 'recurrent',
    signature: [['input_size', REQUIRED], ['hidden_size', REQUIRED], ['num_layers', 1], ['bias', true], ['batch_first', false], ['dropout', 0], ['bidirectional', false], ['nonlinearity', 'tanh'], ['proj_size', 0]],
    build: (a, className) => {
      const input = toInt(a.input_size, 'input_size')
      const hidden = toInt(a.hidden_size, 'hidden_size')
      const layers = toInt(a.num_layers, 'num_layers')
      const directions = a.bidirectional ? 2 : 1
      const params = new Map<string, Tensor>()
      for (let layer = 0; layer < layers; layer++) {
        for (let direction = 0; direction < directions; direction++) {
          const suffix = `l${layer}${direction ? '_reverse' : ''}`
          const layerInput = layer === 0 ? input : hidden * directions
          params.set(`weight_ih_${suffix}`, param(gates * hidden, layerInput))
          params.set(`weight_hh_${suffix}`, param(gates * hidden, hidden))
          if (a.bias) { params.set(`bias_ih_${suffix}`, param(gates * hidden)); params.set(`bias_hh_${suffix}`, param(gates * hidden)) }
        }
      }
      return {
        config: { input_size: input, hidden_size: hidden, num_layers: layers, batch_first: Boolean(a.batch_first), bidirectional: Boolean(a.bidirectional) },
        params, props: new Map<string, Value>([['hidden_size', hidden], ['input_size', input]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape[x.shape.length - 1] !== input) throw new PyError(`nn.${className}(input_size=${input}) received features of size ${x.shape[x.shape.length - 1]}`)
          const output = new Tensor([...x.shape.slice(0, -1), hidden * directions])
          const batch = x.shape.length === 3 ? [a.batch_first ? x.shape[0] : x.shape[1]] : []
          const state = new Tensor([layers * directions, ...batch, hidden])
          return new PyTuple([output, gates === 4 ? new PyTuple([state, new Tensor(state.shape)]) : state])
        },
      }
    },
  }
}

function transformerLayerDefinition(decoder: boolean): LayerDefinition {
  return {
    type: 'attention',
    signature: [['d_model', REQUIRED], ['nhead', REQUIRED], ['dim_feedforward', 2048], ['dropout', 0.1], ['activation', 'relu'], ['layer_norm_eps', 1e-5], ['batch_first', false], ['norm_first', false], ['bias', true]],
    build: (a, className) => {
      const d = toInt(a.d_model, 'd_model')
      const heads = toInt(a.nhead, 'nhead')
      const ff = toInt(a.dim_feedforward, 'dim_feedforward')
      if (d % heads) throw new PyError(`d_model=${d} must be divisible by nhead=${heads}`)
      const attentionCount = decoder ? 2 : 1
      const bias = Boolean(a.bias)
      const params = new Map<string, Tensor>()
      const add = (name: string, tensor: Tensor, isBias = false) => { if (!isBias || bias) params.set(name, tensor) }
      const attentionNames = decoder ? ['self_attn', 'multihead_attn'] : ['self_attn']
      for (const attention of attentionNames) {
        add(`${attention}.in_proj_weight`, param(3 * d, d)); add(`${attention}.in_proj_bias`, param(3 * d), true)
        add(`${attention}.out_proj.weight`, param(d, d)); add(`${attention}.out_proj.bias`, param(d), true)
      }
      add('linear1.weight', param(ff, d)); add('linear1.bias', param(ff), true)
      add('linear2.weight', param(d, ff)); add('linear2.bias', param(d), true)
      for (let index = 1; index <= attentionCount + 1; index++) { add(`norm${index}.weight`, param(d)); add(`norm${index}.bias`, param(d), true) }
      return {
        config: { d_model: d, nhead: heads, dim_feedforward: ff, batch_first: Boolean(a.batch_first), norm_first: Boolean(a.norm_first) },
        params, props: new Map<string, Value>([['d_model', d]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape[x.shape.length - 1] !== d) throw new PyError(`nn.${className}(d_model=${d}) received features of size ${x.shape[x.shape.length - 1]}`)
          return new Tensor(x.shape)
        },
      }
    },
  }
}

/** Shape of a loss: a scalar, or element-wise when reduction='none'. */
export function lossShape(kind: string, inputs: Value[], reduction: Value): number[] {
  const tensors = inputs.filter((item): item is Tensor => item instanceof Tensor)
  if (reduction !== 'none' || !tensors.length) return []
  const [input, target] = tensors
  if ((kind === 'cross_entropy' || kind === 'nll_loss') && target && target.dtype !== 'float') return target.shape
  if (kind === 'cross_entropy' || kind === 'nll_loss') return [input.shape[0], ...input.shape.slice(2)]
  return tensors.reduce((shape, tensor) => broadcast(shape, tensor.shape, kind), [] as number[])
}

function lossDefinition(kind: string, signature: Signature): LayerDefinition {
  return {
    type: 'loss', signature,
    build: (a) => ({
      config: (a.reduction !== undefined && a.reduction !== 'mean' ? { reduction: String(a.reduction) } : {}) as Record<string, string>, params: new Map(), props: new Map(),
      forward: (inputs) => new Tensor(lossShape(kind, inputs, a.reduction ?? 'mean')),
    }),
  }
}

function padDefinition(dims: number, mode: string): LayerDefinition {
  return {
    type: 'reshape', signature: mode === 'constant' ? [['padding', REQUIRED], ['value', 0]] : [['padding', REQUIRED]],
    build: (a, className) => {
      const pad = nTuple(a.padding, 2 * dims, 'padding')
      return {
        config: { padding: pad, mode }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const shape = [...x.shape]
          for (let index = 0; index < dims; index++) shape[shape.length - 1 - index] += pad[2 * index] + pad[2 * index + 1]
          return new Tensor(shape)
        },
      }
    },
  }
}

function cellDefinition(gates: number): LayerDefinition {
  return {
    type: 'recurrent',
    signature: [['input_size', REQUIRED], ['hidden_size', REQUIRED], ['bias', true], ['nonlinearity', 'tanh']],
    build: (a, className) => {
      const input = toInt(a.input_size, 'input_size')
      const hidden = toInt(a.hidden_size, 'hidden_size')
      const params = new Map([['weight_ih', param(gates * hidden, input)], ['weight_hh', param(gates * hidden, hidden)]])
      if (a.bias) { params.set('bias_ih', param(gates * hidden)); params.set('bias_hh', param(gates * hidden)) }
      return {
        config: { input_size: input, hidden_size: hidden }, params, props: new Map<string, Value>([['hidden_size', hidden], ['input_size', input]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape[x.shape.length - 1] !== input) throw new PyError(`nn.${className}(input_size=${input}) received features of size ${x.shape[x.shape.length - 1]}`)
          const state = new Tensor([...x.shape.slice(0, -1), hidden])
          return gates === 4 ? new PyTuple([state, new Tensor(state.shape)]) : state
        },
      }
    },
  }
}

const definitions: Record<string, LayerDefinition> = {
  Linear: {
    type: 'linear',
    signature: [['in_features', REQUIRED], ['out_features', REQUIRED], ['bias', true]],
    build: (a, className) => {
      const inFeatures = toInt(a.in_features, 'in_features')
      const outFeatures = toInt(a.out_features, 'out_features')
      const params = new Map([['weight', param(outFeatures, inFeatures)]])
      if (a.bias) params.set('bias', param(outFeatures))
      return {
        config: { in_features: inFeatures, out_features: outFeatures, ...(a.bias ? {} : { bias: false }) }, params,
        props: new Map<string, Value>([['in_features', inFeatures], ['out_features', outFeatures], ['weight', params.get('weight')!], ['bias', params.get('bias') ?? null]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape[x.shape.length - 1] !== inFeatures) throw new PyError(`nn.Linear(in_features=${inFeatures}) received an input whose last dimension is ${x.shape[x.shape.length - 1]} (shape ${fmt(x.shape)})`)
          return new Tensor([...x.shape.slice(0, -1), outFeatures])
        },
      }
    },
  },
  Conv1d: convDefinition(1), Conv2d: convDefinition(2), Conv3d: convDefinition(3),
  ConvTranspose1d: transposedConvDefinition(1), ConvTranspose2d: transposedConvDefinition(2), ConvTranspose3d: transposedConvDefinition(3),
  BatchNorm1d: normDefinition('batch', 1), BatchNorm2d: normDefinition('batch', 2), BatchNorm3d: normDefinition('batch', 3),
  InstanceNorm1d: normDefinition('instance', 1), InstanceNorm2d: normDefinition('instance', 2), InstanceNorm3d: normDefinition('instance', 3),
  LayerNorm: {
    type: 'norm',
    signature: [['normalized_shape', REQUIRED], ['eps', 1e-5], ['elementwise_affine', true], ['bias', true]],
    build: (a, className) => {
      const shape = toInts(a.normalized_shape, 'normalized_shape')
      const params = new Map<string, Tensor>()
      if (a.elementwise_affine) { params.set('weight', param(...shape)); if (a.bias) params.set('bias', param(...shape)) }
      return {
        config: { normalized_shape: shape.length === 1 ? shape[0] : shape }, params, props: new Map<string, Value>([['weight', params.get('weight') ?? null]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (fmt(x.shape.slice(-shape.length)) !== fmt(shape)) throw new PyError(`nn.LayerNorm(${shape.join(', ')}) expects trailing dimensions ${fmt(shape)}, got ${fmt(x.shape)}`)
          return new Tensor(x.shape)
        },
      }
    },
  },
  RMSNorm: {
    type: 'norm',
    signature: [['normalized_shape', REQUIRED], ['eps', null], ['elementwise_affine', true]],
    build: (a, className) => {
      const shape = toInts(a.normalized_shape, 'normalized_shape')
      const params = new Map<string, Tensor>()
      if (a.elementwise_affine) params.set('weight', param(...shape))
      return {
        config: { normalized_shape: shape.length === 1 ? shape[0] : shape }, params, props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (fmt(x.shape.slice(-shape.length)) !== fmt(shape)) throw new PyError(`nn.RMSNorm(${shape.join(', ')}) expects trailing dimensions ${fmt(shape)}, got ${fmt(x.shape)}`)
          return new Tensor(x.shape)
        },
      }
    },
  },
  GroupNorm: {
    type: 'norm',
    signature: [['num_groups', REQUIRED], ['num_channels', REQUIRED], ['eps', 1e-5], ['affine', true]],
    build: (a, className) => {
      const groups = toInt(a.num_groups, 'num_groups')
      const channels = toInt(a.num_channels, 'num_channels')
      if (channels % groups) throw new PyError(`nn.GroupNorm: num_channels=${channels} must be divisible by num_groups=${groups}`)
      const params = new Map<string, Tensor>()
      if (a.affine) { params.set('weight', param(channels)); params.set('bias', param(channels)) }
      return {
        config: { num_groups: groups, num_channels: channels }, params, props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.shape[1] !== channels) throw new PyError(`nn.GroupNorm(num_channels=${channels}) received ${x.shape[1]} channels (input ${fmt(x.shape)})`)
          return new Tensor(x.shape)
        },
      }
    },
  },
  Embedding: {
    type: 'embedding',
    signature: [['num_embeddings', REQUIRED], ['embedding_dim', REQUIRED], ['padding_idx', null], ['max_norm', null], ['norm_type', 2], ['scale_grad_by_freq', false], ['sparse', false]],
    build: (a, className) => {
      const count = toInt(a.num_embeddings, 'num_embeddings')
      const dim = toInt(a.embedding_dim, 'embedding_dim')
      const params = new Map([['weight', param(count, dim)]])
      return {
        config: { num_embeddings: count, embedding_dim: dim }, params,
        props: new Map<string, Value>([['num_embeddings', count], ['embedding_dim', dim], ['weight', params.get('weight')!]]),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          if (x.dtype === 'float') throw new PyError(`nn.Embedding expects integer token ids, got a float tensor ${fmt(x.shape)}. Use torch.randint(...) for the example input.`)
          return new Tensor([...x.shape, dim])
        },
      }
    },
  },
  Dropout: activation('dropout', [['p', 0.5], ['inplace', false]]), Dropout1d: activation('dropout', [['p', 0.5], ['inplace', false]]),
  Dropout2d: activation('dropout', [['p', 0.5], ['inplace', false]]), Dropout3d: activation('dropout', [['p', 0.5], ['inplace', false]]),
  ReLU: activation('activation', [['inplace', false]]), ReLU6: activation('activation', [['inplace', false]]), GELU: activation('activation', [['approximate', 'none']]),
  SiLU: activation('activation', [['inplace', false]]), Mish: activation('activation', [['inplace', false]]), Tanh: activation(), Sigmoid: activation(),
  LeakyReLU: activation('activation', [['negative_slope', 0.01], ['inplace', false]]), ELU: activation('activation', [['alpha', 1], ['inplace', false]]),
  Softplus: activation('activation', [['beta', 1], ['threshold', 20]]), Hardswish: activation('activation', [['inplace', false]]), Hardsigmoid: activation('activation', [['inplace', false]]),
  Softmax: activation('activation', [['dim', null]]), LogSoftmax: activation('activation', [['dim', null]]),
  Identity: { type: 'generic', signature: [], build: (_, className) => ({ config: {}, params: new Map(), props: new Map(), forward: identityForward(className) }) },
  PReLU: {
    type: 'activation', signature: [['num_parameters', 1], ['init', 0.25]],
    build: (a, className) => ({ config: { num_parameters: toInt(a.num_parameters, 'num_parameters') }, params: new Map([['weight', param(toInt(a.num_parameters, 'num_parameters'))]]), props: new Map(), forward: identityForward(className) }),
  },
  MaxPool1d: poolDefinition('max', 1), MaxPool2d: poolDefinition('max', 2), MaxPool3d: poolDefinition('max', 3),
  AvgPool1d: poolDefinition('avg', 1), AvgPool2d: poolDefinition('avg', 2), AvgPool3d: poolDefinition('avg', 3),
  AdaptiveAvgPool1d: adaptivePoolDefinition(1), AdaptiveAvgPool2d: adaptivePoolDefinition(2), AdaptiveAvgPool3d: adaptivePoolDefinition(3),
  AdaptiveMaxPool1d: adaptivePoolDefinition(1), AdaptiveMaxPool2d: adaptivePoolDefinition(2),
  Flatten: {
    type: 'reshape', signature: [['start_dim', 1], ['end_dim', -1]],
    build: (a, className) => ({
      config: { start_dim: toInt(a.start_dim, 'start_dim'), end_dim: toInt(a.end_dim, 'end_dim') }, params: new Map(), props: new Map(),
      forward: (inputs) => new Tensor(flatten(tensorArg(inputs[0], className).shape, toInt(a.start_dim, 'start_dim'), toInt(a.end_dim, 'end_dim'))),
    }),
  },
  Unflatten: {
    type: 'reshape', signature: [['dim', REQUIRED], ['unflattened_size', REQUIRED]],
    build: (a, className) => {
      const sizes = toInts(a.unflattened_size, 'unflattened_size')
      return {
        config: { dim: toInt(a.dim, 'dim'), unflattened_size: sizes }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const dim = normalizeDim(toInt(a.dim, 'dim'), x.shape.length)
          if (sizes.reduce((p, s) => p * s, 1) !== x.shape[dim]) throw new PyError(`nn.Unflatten: ${sizes.join('×')} does not equal dimension size ${x.shape[dim]}`)
          return new Tensor([...x.shape.slice(0, dim), ...sizes, ...x.shape.slice(dim + 1)])
        },
      }
    },
  },
  Upsample: {
    type: 'upsample', signature: [['size', null], ['scale_factor', null], ['mode', 'nearest'], ['align_corners', null], ['recompute_scale_factor', null]],
    build: (a, className) => ({
      config: { ...(a.scale_factor !== null ? { scale_factor: formatValue(a.scale_factor) } : {}), ...(a.size !== null ? { size: formatValue(a.size) } : {}), mode: String(a.mode) }, params: new Map(), props: new Map(),
      forward: (inputs) => {
        const x = tensorArg(inputs[0], className)
        return new Tensor(interpolateShape(x.shape, a.size, a.scale_factor))
      },
    }),
  },
  PixelShuffle: {
    type: 'upsample', signature: [['upscale_factor', REQUIRED]],
    build: (a, className) => {
      const r = toInt(a.upscale_factor, 'upscale_factor')
      return {
        config: { upscale_factor: r }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const [c, h, w] = x.shape.slice(-3)
          if (c % (r * r)) throw new PyError(`nn.PixelShuffle(${r}) needs channels divisible by ${r * r}, got ${c}`)
          return new Tensor([...x.shape.slice(0, -3), c / (r * r), h * r, w * r])
        },
      }
    },
  },
  MultiheadAttention: {
    type: 'attention',
    signature: [['embed_dim', REQUIRED], ['num_heads', REQUIRED], ['dropout', 0], ['bias', true], ['add_bias_kv', false], ['add_zero_attn', false], ['kdim', null], ['vdim', null], ['batch_first', false]],
    build: (a, className) => {
      const e = toInt(a.embed_dim, 'embed_dim')
      const heads = toInt(a.num_heads, 'num_heads')
      if (e % heads) throw new PyError(`embed_dim=${e} must be divisible by num_heads=${heads}`)
      const params = new Map([['in_proj_weight', param(3 * e, e)], ['out_proj.weight', param(e, e)]])
      if (a.bias) { params.set('in_proj_bias', param(3 * e)); params.set('out_proj.bias', param(e)) }
      const batchFirst = Boolean(a.batch_first)
      return {
        config: { embed_dim: e, num_heads: heads, head_dim: e / heads, batch_first: batchFirst }, params, props: new Map<string, Value>([['embed_dim', e], ['num_heads', heads]]),
        forward: (inputs, kwargs) => {
          const query = tensorArg(inputs[0] ?? kwargs.get('query'), className)
          const key = tensorArg(inputs[1] ?? kwargs.get('key') ?? query, className)
          if (query.shape[query.shape.length - 1] !== e) throw new PyError(`nn.MultiheadAttention(embed_dim=${e}) received features of size ${query.shape[query.shape.length - 1]}`)
          const batched = query.shape.length === 3
          const l = batched ? query.shape[batchFirst ? 1 : 0] : query.shape[0]
          const s = batched ? key.shape[batchFirst ? 1 : 0] : key.shape[0]
          const n = batched ? query.shape[batchFirst ? 0 : 1] : 1
          return new PyTuple([new Tensor(query.shape), new Tensor(batched ? [n, l, s] : [l, s])])
        },
      }
    },
  },
  EmbeddingBag: {
    type: 'embedding',
    signature: [['num_embeddings', REQUIRED], ['embedding_dim', REQUIRED], ['max_norm', null], ['norm_type', 2], ['scale_grad_by_freq', false], ['mode', 'mean'], ['sparse', false], ['_weight', null], ['include_last_offset', false], ['padding_idx', null]],
    build: (a, className) => {
      const count = toInt(a.num_embeddings, 'num_embeddings')
      const dim = toInt(a.embedding_dim, 'embedding_dim')
      const params = new Map([['weight', param(count, dim)]])
      return {
        config: { num_embeddings: count, embedding_dim: dim, mode: String(a.mode) }, params,
        props: new Map<string, Value>([['num_embeddings', count], ['embedding_dim', dim], ['weight', params.get('weight')!]]),
        forward: (inputs, kwargs) => {
          const x = tensorArg(inputs[0], className)
          if (x.dtype === 'float') throw new PyError(`nn.EmbeddingBag expects integer token ids, got a float tensor ${fmt(x.shape)}. Use torch.randint(...) for the example input.`)
          if (x.shape.length === 2) return new Tensor([x.shape[0], dim])
          const offsets = inputs[1] ?? kwargs.get('offsets')
          const bags = offsets instanceof Tensor ? offsets.shape[0] - (a.include_last_offset ? 1 : 0) : 1
          return new Tensor([bags, dim])
        },
      }
    },
  },
  TransformerEncoderLayer: transformerLayerDefinition(false),
  TransformerDecoderLayer: transformerLayerDefinition(true),
  LSTM: recurrentDefinition(4), GRU: recurrentDefinition(3), RNN: recurrentDefinition(1),
  ZeroPad2d: {
    type: 'reshape', signature: [['padding', REQUIRED]],
    build: (a, className) => {
      const pad = nTuple(a.padding, 4, 'padding')
      return {
        config: { padding: pad }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const shape = [...x.shape]
          shape[shape.length - 1] += pad[0] + pad[1]
          shape[shape.length - 2] += pad[2] + pad[3]
          return new Tensor(shape)
        },
      }
    },
  },
  MSELoss: lossDefinition('mse_loss', [['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  L1Loss: lossDefinition('l1_loss', [['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  SmoothL1Loss: lossDefinition('smooth_l1_loss', [['size_average', null], ['reduce', null], ['reduction', 'mean'], ['beta', 1]]),
  HuberLoss: lossDefinition('huber_loss', [['reduction', 'mean'], ['delta', 1]]),
  CrossEntropyLoss: lossDefinition('cross_entropy', [['weight', null], ['size_average', null], ['ignore_index', -100], ['reduce', null], ['reduction', 'mean'], ['label_smoothing', 0]]),
  NLLLoss: lossDefinition('nll_loss', [['weight', null], ['size_average', null], ['ignore_index', -100], ['reduce', null], ['reduction', 'mean']]),
  BCELoss: lossDefinition('binary_cross_entropy', [['weight', null], ['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  BCEWithLogitsLoss: lossDefinition('binary_cross_entropy_with_logits', [['weight', null], ['size_average', null], ['reduce', null], ['reduction', 'mean'], ['pos_weight', null]]),
  KLDivLoss: lossDefinition('kl_div', [['size_average', null], ['reduce', null], ['reduction', 'mean'], ['log_target', false]]),
  CosineEmbeddingLoss: lossDefinition('cosine_embedding_loss', [['margin', 0], ['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  MarginRankingLoss: lossDefinition('margin_ranking_loss', [['margin', 0], ['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  TripletMarginLoss: lossDefinition('triplet_margin_loss', [['margin', 1], ['p', 2], ['eps', 1e-6], ['swap', false], ['size_average', null], ['reduce', null], ['reduction', 'mean']]),
  SyncBatchNorm: normDefinition('batch', 2),
  ReflectionPad1d: padDefinition(1, 'reflect'), ReflectionPad2d: padDefinition(2, 'reflect'), ReflectionPad3d: padDefinition(3, 'reflect'),
  ReplicationPad1d: padDefinition(1, 'replicate'), ReplicationPad2d: padDefinition(2, 'replicate'), ReplicationPad3d: padDefinition(3, 'replicate'),
  ConstantPad1d: padDefinition(1, 'constant'), ConstantPad2d: padDefinition(2, 'constant'), ConstantPad3d: padDefinition(3, 'constant'),
  ZeroPad1d: padDefinition(1, 'zeros'), ZeroPad3d: padDefinition(3, 'zeros'),
  LSTMCell: cellDefinition(4), GRUCell: cellDefinition(3), RNNCell: cellDefinition(1),
  Hardtanh: activation('activation', [['min_val', -1], ['max_val', 1], ['inplace', false]]), SELU: activation('activation', [['inplace', false]]),
  CELU: activation('activation', [['alpha', 1], ['inplace', false]]), Softsign: activation(), Softmin: activation('activation', [['dim', null]]),
  LogSigmoid: activation(), Tanhshrink: activation(), Hardshrink: activation('activation', [['lambd', 0.5]]), Softshrink: activation('activation', [['lambd', 0.5]]),
  Threshold: activation('activation', [['threshold', REQUIRED], ['value', REQUIRED], ['inplace', false]]), Softmax2d: activation(),
  AlphaDropout: activation('dropout', [['p', 0.5], ['inplace', false]]), FeatureAlphaDropout: activation('dropout', [['p', 0.5], ['inplace', false]]),
  LocalResponseNorm: activation('norm', [['size', REQUIRED], ['alpha', 1e-4], ['beta', 0.75], ['k', 1]]),
  GLU: {
    type: 'activation', signature: [['dim', -1]],
    build: (a, className) => ({
      config: { dim: toInt(a.dim, 'dim') }, params: new Map(), props: new Map(),
      forward: (inputs) => {
        const x = tensorArg(inputs[0], className)
        const axis = normalizeDim(toInt(a.dim, 'dim'), x.shape.length)
        if (x.shape[axis] % 2) throw new PyError(`nn.GLU needs an even size along dim ${toInt(a.dim, 'dim')}, got ${x.shape[axis]}`)
        const shape = [...x.shape]
        shape[axis] /= 2
        return new Tensor(shape)
      },
    }),
  },
  Bilinear: {
    type: 'linear', signature: [['in1_features', REQUIRED], ['in2_features', REQUIRED], ['out_features', REQUIRED], ['bias', true]],
    build: (a, className) => {
      const [in1, in2, out] = [toInt(a.in1_features, 'in1_features'), toInt(a.in2_features, 'in2_features'), toInt(a.out_features, 'out_features')]
      const params = new Map([['weight', param(out, in1, in2)]])
      if (a.bias) params.set('bias', param(out))
      return {
        config: { in1_features: in1, in2_features: in2, out_features: out }, params, props: new Map<string, Value>([['out_features', out]]),
        forward: (inputs) => { const x = tensorArg(inputs[0], className); return new Tensor([...x.shape.slice(0, -1), out]) },
      }
    },
  },
  CosineSimilarity: {
    type: 'generic', signature: [['dim', 1], ['eps', 1e-8]],
    build: (a, className) => ({
      config: { dim: toInt(a.dim, 'dim') }, params: new Map(), props: new Map(),
      forward: (inputs) => {
        const shape = broadcast(tensorArg(inputs[0], className).shape, tensorArg(inputs[1], className).shape, 'CosineSimilarity')
        return new Tensor(shape.filter((_, index) => index !== normalizeDim(toInt(a.dim, 'dim'), shape.length)))
      },
    }),
  },
  PairwiseDistance: {
    type: 'generic', signature: [['p', 2], ['eps', 1e-6], ['keepdim', false]],
    build: (a, className) => ({
      config: {}, params: new Map(), props: new Map(),
      forward: (inputs) => {
        const shape = broadcast(tensorArg(inputs[0], className).shape, tensorArg(inputs[1], className).shape, 'PairwiseDistance')
        return new Tensor(a.keepdim ? [...shape.slice(0, -1), 1] : shape.slice(0, -1))
      },
    }),
  },
  PixelUnshuffle: {
    type: 'downsample', signature: [['downscale_factor', REQUIRED]],
    build: (a, className) => {
      const r = toInt(a.downscale_factor, 'downscale_factor')
      return {
        config: { downscale_factor: r }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const [c, h, w] = x.shape.slice(-3)
          if (h % r || w % r) throw new PyError(`nn.PixelUnshuffle(${r}) needs height and width divisible by ${r}, got ${h}×${w}`)
          return new Tensor([...x.shape.slice(0, -3), c * r * r, h / r, w / r])
        },
      }
    },
  },
  ChannelShuffle: activation('generic', [['groups', REQUIRED]]),
  UpsamplingNearest2d: {
    type: 'upsample', signature: [['size', null], ['scale_factor', null]],
    build: (a, className) => ({ config: { mode: 'nearest' }, params: new Map(), props: new Map(), forward: (inputs) => new Tensor(interpolateShape(tensorArg(inputs[0], className).shape, a.size, a.scale_factor)) }),
  },
  UpsamplingBilinear2d: {
    type: 'upsample', signature: [['size', null], ['scale_factor', null]],
    build: (a, className) => ({ config: { mode: 'bilinear' }, params: new Map(), props: new Map(), forward: (inputs) => new Tensor(interpolateShape(tensorArg(inputs[0], className).shape, a.size, a.scale_factor)) }),
  },
  Unfold: {
    type: 'reshape', signature: [['kernel_size', REQUIRED], ['dilation', 1], ['padding', 0], ['stride', 1]],
    build: (a, className) => {
      const kernel = nTuple(a.kernel_size, 2, 'kernel_size')
      const dilation = nTuple(a.dilation, 2, 'dilation')
      const padding = nTuple(a.padding, 2, 'padding')
      const stride = nTuple(a.stride, 2, 'stride')
      return {
        config: { kernel_size: kernel[0], stride: stride[0] }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const [n, c, h, w] = x.shape.length === 4 ? x.shape : [null, ...x.shape]
          const blocks = convAxis(h as number, kernel[0], stride[0], padding[0], dilation[0]) * convAxis(w as number, kernel[1], stride[1], padding[1], dilation[1])
          return new Tensor([...(n === null ? [] : [n]), (c as number) * kernel[0] * kernel[1], blocks])
        },
      }
    },
  },
  Fold: {
    type: 'reshape', signature: [['output_size', REQUIRED], ['kernel_size', REQUIRED], ['dilation', 1], ['padding', 0], ['stride', 1]],
    build: (a, className) => {
      const size = nTuple(a.output_size, 2, 'output_size')
      const kernel = nTuple(a.kernel_size, 2, 'kernel_size')
      return {
        config: { output_size: size, kernel_size: kernel[0] }, params: new Map(), props: new Map(),
        forward: (inputs) => {
          const x = tensorArg(inputs[0], className)
          const channels = x.shape[x.shape.length - 2] / (kernel[0] * kernel[1])
          return new Tensor([...x.shape.slice(0, -2), channels, ...size])
        },
      }
    },
  },
}

export function interpolateShape(shape: number[], size: Value, scale: Value): number[] {
  const spatial = Math.max(1, shape.length - 2)
  if (size !== null && size !== undefined) return [...shape.slice(0, 2), ...nTuple(size, spatial, 'size')]
  if (scale === null || scale === undefined) throw new PyError('Upsample/interpolate needs size or scale_factor')
  const factors = scale instanceof PyTuple || scale instanceof PyList ? scale.items.map((item) => toNumber(item, 'scale_factor')) : Array(spatial).fill(toNumber(scale, 'scale_factor'))
  return [...shape.slice(0, 2), ...shape.slice(2).map((value, index) => Math.floor(value * (factors[index] ?? 1)))]
}

export const supportedLayers = Object.keys(definitions)

export function isLayer(name: string): boolean { return name in definitions }

export function layerType(name: string): ModelNodeType { return definitions[name]?.type ?? 'generic' }

export function createLayer(className: string, args: Value[], kwargs: Map<string, Value>): LayerConfig {
  const definition = definitions[className]
  if (!definition) throw new PyError(`nn.${className} is not supported by the Atlas tracer yet`)
  const bound = bind(className, definition.signature, args, kwargs)
  return { className, ...definition.build(bound, className) }
}

export function dictValue(value: Value): Record<string, Value> {
  if (!(value instanceof PyDict)) return {}
  return Object.fromEntries([...value.map.entries()].map(([key, item]) => [String(key), item]))
}

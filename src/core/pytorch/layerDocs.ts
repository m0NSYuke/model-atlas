// Bilingual learning notes generated for traced layers and operations.
import type { TraceNode } from './interpreter'

interface Doc { en: string; zh: string; why?: [string, string]; formula?: string; zhName?: string }

const docs: Record<string, Doc> = {
  Linear: { en: 'Fully connected layer: every output feature is a weighted sum of all input features.', zh: '全连接层：每个输出特征都是全部输入特征的加权和。', why: ['Mixes features and changes the feature dimension.', '混合特征并改变特征维度。'], formula: 'y = x·Wᵀ + b', zhName: '全连接层' },
  Conv1d: { en: '1D convolution slides a learned kernel along a sequence.', zh: '一维卷积沿序列滑动可学习卷积核。', formula: 'Lout = ⌊(L + 2p − d(k−1) − 1)/s⌋ + 1', zhName: '一维卷积' },
  Conv2d: { en: '2D convolution slides learned k×k filters over the feature map; each filter produces one output channel.', zh: '二维卷积在特征图上滑动 k×k 卷积核，每个卷积核产生一个输出通道。', why: ['Shares weights across positions, so it detects the same pattern anywhere in the image.', '在不同位置共享权重，因此能在图像任何位置识别同一种模式。'], formula: 'Hout = ⌊(H + 2p − d(k−1) − 1)/s⌋ + 1', zhName: '二维卷积' },
  Conv3d: { en: '3D convolution filters over time, height and width at once.', zh: '三维卷积同时在时间、高度和宽度上滤波。', formula: 'Tout = ⌊(T + 2p − k)/s⌋ + 1', zhName: '三维卷积' },
  ConvTranspose1d: { en: 'Transposed convolution upsamples a sequence with a learned kernel.', zh: '转置卷积用可学习卷积核对序列上采样。', zhName: '转置卷积' },
  ConvTranspose2d: { en: 'Transposed convolution (“deconvolution”) upsamples a feature map with learned weights.', zh: '转置卷积（“反卷积”）用可学习权重放大特征图。', formula: 'Hout = (H − 1)·s − 2p + k + output_padding', zhName: '转置卷积' },
  ConvTranspose3d: { en: 'Transposed 3D convolution upsamples a video volume.', zh: '三维转置卷积放大视频体积。', zhName: '三维转置卷积' },
  BatchNorm1d: { en: 'Batch normalization standardizes each channel with batch statistics, then applies a learned scale and shift.', zh: '批归一化用批次统计量标准化每个通道，再进行可学习的缩放和平移。', formula: 'y = γ·(x − μ_B)/√(σ²_B + ε) + β', zhName: '批归一化' },
  BatchNorm2d: { en: 'Batch normalization standardizes each channel over (N, H, W), then applies learned γ and β.', zh: '批归一化在 (N, H, W) 上标准化每个通道，再应用可学习的 γ 和 β。', why: ['Stabilizes and speeds up training of deep CNNs.', '让深层 CNN 训练更稳定、更快。'], formula: 'y = γ·(x − μ_B)/√(σ²_B + ε) + β', zhName: '批归一化' },
  BatchNorm3d: { en: 'Batch normalization over (N, T, H, W) per channel.', zh: '按通道在 (N, T, H, W) 上做批归一化。', zhName: '批归一化' },
  LayerNorm: { en: 'Layer normalization standardizes each token’s features independently of the batch.', zh: '层归一化独立地标准化每个 token 的特征，与批次无关。', why: ['Keeps activations well scaled in Transformers; works with any batch size.', '让 Transformer 的激活尺度稳定，且不依赖批次大小。'], formula: 'y = γ·(x − mean)/√(var + ε) + β', zhName: '层归一化' },
  RMSNorm: { en: 'RMS normalization rescales features by their root mean square (no mean subtraction).', zh: 'RMS 归一化按均方根缩放特征（不减均值）。', formula: 'y = γ·x / √(mean(x²) + ε)', zhName: 'RMS 归一化' },
  GroupNorm: { en: 'Group normalization standardizes groups of channels per sample.', zh: '组归一化在每个样本内按通道分组标准化。', why: ['Works with small batches, common in diffusion U-Nets.', '小批次下也有效，常用于扩散 U-Net。'], zhName: '组归一化' },
  Embedding: { en: 'Lookup table: each integer token id selects one learned vector.', zh: '查找表：每个整数 token id 取出一条可学习向量。', formula: 'E[id] ∈ ℝᵈ', zhName: '嵌入层' },
  Dropout: { en: 'Randomly zeros activations during training (identity at inference).', zh: '训练时随机将激活置零（推理时为恒等映射）。', why: ['Regularizes the network to reduce overfitting.', '正则化网络、减少过拟合。'], zhName: '随机失活' },
  ReLU: { en: 'Rectified linear unit keeps positive values and zeros negatives.', zh: 'ReLU 保留正值、将负值置零。', formula: 'ReLU(x) = max(0, x)', zhName: 'ReLU 激活' },
  GELU: { en: 'Gaussian error linear unit, a smooth ReLU used in Transformers.', zh: 'GELU 是平滑版 ReLU，常用于 Transformer。', formula: 'GELU(x) = x·Φ(x)', zhName: 'GELU 激活' },
  SiLU: { en: 'SiLU / Swish activation.', zh: 'SiLU / Swish 激活函数。', formula: 'SiLU(x) = x·σ(x)', zhName: 'SiLU 激活' },
  Tanh: { en: 'Hyperbolic tangent squashes values into (−1, 1).', zh: '双曲正切把数值压缩到 (−1, 1)。', zhName: 'Tanh 激活' },
  Sigmoid: { en: 'Sigmoid squashes values into (0, 1).', zh: 'Sigmoid 把数值压缩到 (0, 1)。', formula: 'σ(x) = 1/(1 + e⁻ˣ)', zhName: 'Sigmoid 激活' },
  LeakyReLU: { en: 'ReLU with a small slope for negative values.', zh: '负半轴保留小斜率的 ReLU。', zhName: 'LeakyReLU 激活' },
  Softmax: { en: 'Turns scores into a probability distribution along one dimension.', zh: '沿某一维把分数变成概率分布。', formula: 'softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ', zhName: 'Softmax' },
  MaxPool2d: { en: 'Max pooling keeps the largest value in each window, shrinking H and W.', zh: '最大池化保留每个窗口中的最大值，缩小 H 和 W。', zhName: '最大池化' },
  AvgPool2d: { en: 'Average pooling averages each window, shrinking H and W.', zh: '平均池化对每个窗口取平均，缩小 H 和 W。', zhName: '平均池化' },
  AdaptiveAvgPool2d: { en: 'Averages to a fixed output size regardless of input resolution.', zh: '无论输入分辨率如何，都平均池化到固定输出尺寸。', why: ['Lets the classifier accept images of any size.', '让分类头可以接受任意尺寸的图像。'], zhName: '自适应平均池化' },
  Flatten: { en: 'Flattens dimensions into one vector per sample.', zh: '将多个维度展平成每个样本的一条向量。', zhName: '展平' },
  Upsample: { en: 'Upsamples by interpolation (no learned weights).', zh: '通过插值上采样（无可学习参数）。', zhName: '上采样' },
  PixelShuffle: { en: 'Rearranges channels into a higher-resolution grid.', zh: '把通道重排成更高分辨率的网格。', zhName: '像素重排' },
  MultiheadAttention: { en: 'Multi-head attention: each token builds queries, keys and values and attends to all tokens in several heads.', zh: '多头注意力：每个 token 生成 Q、K、V，并在多个头中关注所有 token。', why: ['Lets every position mix information from every other position.', '让每个位置都能汇聚其他所有位置的信息。'], formula: 'Attention(Q,K,V) = softmax(QKᵀ/√d)·V', zhName: '多头注意力' },
  TransformerEncoderLayer: { en: 'A complete Transformer encoder layer: self-attention and a feed-forward network, each with residual and LayerNorm.', zh: '完整的 Transformer 编码层：自注意力与前馈网络，各带残差和层归一化。', zhName: 'Transformer 编码层' },
  TransformerDecoderLayer: { en: 'Transformer decoder layer: self-attention, cross-attention and a feed-forward network.', zh: 'Transformer 解码层：自注意力、交叉注意力和前馈网络。', zhName: 'Transformer 解码层' },
  LSTM: { en: 'Long short-term memory: a gated recurrent network that carries a cell state through the sequence.', zh: '长短期记忆网络：带门控、沿序列传递细胞状态的循环网络。', formula: 'cₜ = fₜ⊙cₜ₋₁ + iₜ⊙gₜ ,  hₜ = oₜ⊙tanh(cₜ)', zhName: 'LSTM' },
  GRU: { en: 'Gated recurrent unit: a lighter gated RNN.', zh: '门控循环单元：更轻量的门控 RNN。', zhName: 'GRU' },
  RNN: { en: 'Vanilla recurrent network.', zh: '基础循环神经网络。', zhName: 'RNN' },
  Identity: { en: 'Passes the input through unchanged.', zh: '原样输出输入。', zhName: '恒等映射' },
  // Operations recorded from forward()
  add: { en: 'Element-wise addition. When one input skips over layers, this is a residual connection.', zh: '逐元素相加。若一个输入跳过了若干层，这就是残差连接。', why: ['Residual paths let gradients flow directly and make deep networks trainable.', '残差路径让梯度直接回传，使深层网络可训练。'], formula: 'y = x + F(x)', zhName: '相加' },
  sub: { en: 'Element-wise subtraction.', zh: '逐元素相减。', zhName: '相减' },
  mul: { en: 'Element-wise multiplication (gating or modulation).', zh: '逐元素相乘（门控或调制）。', zhName: '相乘' },
  div: { en: 'Element-wise division.', zh: '逐元素相除。', zhName: '相除' },
  matmul: { en: 'Matrix multiplication over the last two dimensions (batched over the rest).', zh: '在最后两个维度做矩阵乘法（其余维度批量处理）。', formula: '[…, n, k] @ […, k, m] → […, n, m]', zhName: '矩阵乘法' },
  einsum: { en: 'Einstein summation: a general tensor contraction.', zh: '爱因斯坦求和：通用的张量收缩运算。', zhName: '爱因斯坦求和' },
  concat: { en: 'Concatenates tensors along one dimension. In U-Nets this merges a skip connection.', zh: '沿某一维拼接张量。在 U-Net 中用于合并跳跃连接。', zhName: '拼接' },
  stack: { en: 'Stacks tensors along a new dimension.', zh: '沿新维度堆叠张量。', zhName: '堆叠' },
  chunk: { en: 'Splits a tensor into equal chunks.', zh: '将张量均分为若干块。', zhName: '分块' },
  split: { en: 'Splits a tensor into pieces of given sizes (e.g. Q, K and V).', zh: '按给定大小切分张量（例如 Q、K、V）。', zhName: '切分' },
  softmax: { en: 'Normalizes attention scores into weights that sum to 1.', zh: '把注意力分数归一化为和为 1 的权重。', formula: 'softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ', zhName: 'Softmax' },
  attention: { en: 'Scaled dot-product attention (fused kernel).', zh: '缩放点积注意力（融合算子）。', formula: 'softmax(QKᵀ/√d + mask)·V', zhName: '缩放点积注意力' },
  mask: { en: 'Applies a mask: masked positions are set to −∞ so softmax ignores them (causal attention).', zh: '应用掩码：被遮挡位置设为 −∞，softmax 后权重为 0（因果注意力）。', zhName: '掩码' },
  mean: { en: 'Averages over the given dimension (e.g. global pooling over tokens).', zh: '沿给定维度取平均（例如对 token 做全局池化）。', zhName: '求平均' },
  sum: { en: 'Sums over the given dimension.', zh: '沿给定维度求和。', zhName: '求和' },
  interpolate: { en: 'Resizes by interpolation.', zh: '通过插值缩放尺寸。', zhName: '插值缩放' },
  randn_like: { en: 'Samples standard Gaussian noise with the same shape — the ε in the reparameterization trick.', zh: '采样同形状的标准高斯噪声——重参数化技巧中的 ε。', zhName: '高斯噪声 ε' },
  arange: { en: 'Creates position indices 0…N−1.', zh: '生成位置索引 0…N−1。', zhName: '位置索引' },
  exp: { en: 'Element-wise exponential.', zh: '逐元素指数。', zhName: '指数' },
  cos: { en: 'Element-wise cosine (sinusoidal embedding).', zh: '逐元素余弦（正弦位置/时间编码）。', zhName: '余弦' },
  sin: { en: 'Element-wise sine (sinusoidal embedding).', zh: '逐元素正弦（正弦位置/时间编码）。', zhName: '正弦' },
  layer_norm: { en: 'Functional layer normalization.', zh: '函数式层归一化。', zhName: '层归一化' },
  cross_entropy: { en: 'Cross-entropy loss between logits and target classes.', zh: 'logits 与目标类别之间的交叉熵损失。', zhName: '交叉熵损失' },
  mse_loss: { en: 'Mean squared error loss.', zh: '均方误差损失。', zhName: '均方误差' },
  l1_loss: { en: 'Mean absolute error loss.', zh: '平均绝对误差损失。', zhName: 'L1 损失' },
  CrossEntropyLoss: { en: 'Cross-entropy loss module: compares logits with the target classes.', zh: '交叉熵损失模块：比较 logits 与目标类别。', formula: 'ℒ = −Σ y·log softmax(z)', zhName: '交叉熵损失' },
  MSELoss: { en: 'Mean squared error loss module.', zh: '均方误差损失模块。', formula: 'ℒ = mean((ŷ − y)²)', zhName: '均方误差' },
  backward: { en: 'loss.backward(): backpropagation computes the gradient of the loss with respect to every learnable parameter.', zh: 'loss.backward()：反向传播，求出损失对每个可学习参数的梯度。', why: ['Gradients tell the optimizer which direction lowers the loss.', '梯度告诉优化器往哪个方向改参数能让损失变小。'], formula: '∇θ ℒ', zhName: '反向传播' },
  step: { en: 'optimizer.step(): updates every parameter using its gradient. This ends one training step.', zh: 'optimizer.step()：用梯度更新全部参数，一个训练步到此结束。', why: ['Repeating this step over many batches is what training is.', '在大量批次上重复这一步，就是“训练”。'], formula: 'θ ← θ − lr·update(∇θ ℒ)' },
  clip_grad_norm: { en: 'Gradient clipping rescales gradients whose total norm exceeds max_norm.', zh: '梯度裁剪：梯度总范数超过 max_norm 时按比例缩小。', why: ['Prevents a single bad batch from blowing up training.', '防止个别异常批次让训练发散。'], zhName: '梯度裁剪' },
  clip_grad_value: { en: 'Clamps every gradient value into [−clip_value, clip_value].', zh: '把每个梯度值限制在 [−clip_value, clip_value] 内。', zhName: '梯度截断' },
}

const typeDocs: Record<string, Doc> = {
  tensor: { en: 'A tensor flowing through the model.', zh: '在模型中流动的张量。', zhName: '张量' },
  parameter: { en: 'A learnable tensor registered with nn.Parameter (e.g. a class token or positional embedding).', zh: '用 nn.Parameter 注册的可学习张量（如分类 token 或位置编码）。', zhName: '可学习参数' },
  activation: { en: 'Element-wise non-linearity.', zh: '逐元素非线性函数。' },
  pool: { en: 'Reduces a dimension by pooling or reduction.', zh: '通过池化或归约缩小某个维度。' },
  module: { en: 'A user-defined nn.Module that groups the layers inside it.', zh: '用户定义的 nn.Module，包含其中的各层。' },
  attention: { en: 'Attention block.', zh: '注意力模块。' },
  mlp: { en: 'Feed-forward (MLP) block.', zh: '前馈（MLP）模块。' },
}

const fmt = (shape?: number[]) => shape ? `[${shape.join(', ')}]` : '—'

function paramNote(node: TraceNode): [string, string] | undefined {
  if (!node.params) return undefined
  const count = node.params.toLocaleString('en-US')
  const config = node.config ?? {}
  if (node.className === 'Linear') {
    const { in_features: i, out_features: o } = config as { in_features: number; out_features: number }
    const bias = config.bias === false ? '' : ` + ${o}`
    return [`Parameters: ${i}×${o}${bias} = ${count}`, `参数量：${i}×${o}${bias} = ${count}`]
  }
  if (/^Conv[123]d$/.test(node.className)) {
    const { in_channels: i, out_channels: o, kernel_size: k } = config as { in_channels: number; out_channels: number; kernel_size: number | number[] }
    const kernel = Array.isArray(k) ? k.join('×') : node.className === 'Conv1d' ? `${k}` : node.className === 'Conv3d' ? `${k}×${k}×${k}` : `${k}×${k}`
    return [`Parameters: ${o} filters × (${i}×${kernel}) + bias = ${count}`, `参数量：${o} 个卷积核 × (${i}×${kernel}) + 偏置 = ${count}`]
  }
  return [`Learnable parameters: ${count}`, `可学习参数：${count}`]
}

function shapeNote(node: TraceNode): [string, string] | undefined {
  const input = node.inputShapes[0]
  const output = node.outputShapes[0]
  if (!input || !output) return undefined
  if (/^Conv[123]d$/.test(node.className) && node.config) {
    const { kernel_size: k, stride: s, padding: p } = node.config as { kernel_size: number; stride: number; padding: number | string }
    const h = input[input.length - 1]
    const out = output[output.length - 1]
    if (typeof k === 'number' && typeof s === 'number' && typeof p === 'number') {
      return [`Spatial size: ⌊(${h} + 2·${p} − ${k}) / ${s}⌋ + 1 = ${out}`, `空间尺寸：⌊(${h} + 2·${p} − ${k}) / ${s}⌋ + 1 = ${out}`]
    }
  }
  if (fmt(input) !== fmt(output)) return [`Shape ${fmt(input)} → ${fmt(output)}`, `形状 ${fmt(input)} → ${fmt(output)}`]
  return undefined
}

export function describeLayer(node: TraceNode) {
  const doc = docs[node.className] ?? docs[node.op ?? ''] ?? (node.className.startsWith('torch.') ? docs[node.className.slice(6)] : undefined) ?? typeDocs[node.type]
  const params = paramNote(node)
  const shape = shapeNote(node)
  const enNotes = [shape?.[0], params?.[0]].filter(Boolean).join(' · ') || undefined
  const zhNotes = [shape?.[1], params?.[1]].filter(Boolean).join(' · ') || undefined
  const isInput = node.className === 'Input'
  const isOutput = node.className === 'Output'
  const batch = isInput && node.config?.batch_size !== undefined
  const enDescription = batch ? `A batch “${node.name}” from the DataLoader, shape ${fmt(node.outputShapes[0])} (batch_size = ${node.config!.batch_size}, ${node.config!.samples} samples).` : isInput ? `Model input “${node.name}” with shape ${fmt(node.outputShapes[0])}.` : isOutput ? `Model output with shape ${fmt(node.outputShapes[0])}.` : doc?.en ?? `${node.className} operation.`
  const zhDescription = batch ? `DataLoader 取出的一批数据“${node.name}”，形状 ${fmt(node.outputShapes[0])}（batch_size = ${node.config!.batch_size}，共 ${node.config!.samples} 个样本）。` : isInput ? `模型输入“${node.name}”，形状为 ${fmt(node.outputShapes[0])}。` : isOutput ? `模型输出，形状为 ${fmt(node.outputShapes[0])}。` : doc?.zh ?? `${node.className} 运算。`
  const optimizer = node.op === 'step' ? node.className.replace(/\.step$/, '') : undefined
  if (optimizer && node.config?.loop) {
    const [en, zh] = [`Loops ${node.config.loop} = ${Number(node.config.steps).toLocaleString('en-US')} steps; the graph shows one of them.`, `循环 ${node.config.loop} = ${Number(node.config.steps).toLocaleString('en-US')} 步，图中画的是其中一步。`]
    return {
      en: { description: enDescription, learning: { what: doc?.en, why: doc?.why?.[0], notes: en, formula: doc?.formula } },
      zh: { name: `${optimizer} 更新参数`, description: zhDescription, learning: { what: doc?.zh, why: doc?.why?.[1], notes: zh, formula: doc?.formula } },
    }
  }
  const zhName = optimizer ? `${optimizer} 更新参数` : batch ? `批数据 ${node.name}` : isInput ? `输入 ${node.name}` : isOutput ? '输出'
    : !node.group && doc?.zhName && node.op ? doc.zhName
    : !node.group && doc?.zhName && /^\d+$/.test(node.name) ? `${doc.zhName} ${node.name}` : undefined
  return {
    en: { description: enDescription, learning: { what: doc?.en, why: doc?.why?.[0], notes: enNotes, formula: doc?.formula } },
    zh: { name: zhName, description: zhDescription, learning: { what: doc?.zh, why: doc?.why?.[1], notes: zhNotes, formula: doc?.formula } },
  }
}

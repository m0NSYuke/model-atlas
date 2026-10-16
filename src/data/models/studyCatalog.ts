import type { ModelEdge, ModelIR, ModelNode, ModelNodeType, TensorShape } from '../../core/model-ir/types'
import { withDefaultLayout } from '../../core/layout/modelLayout'
import { vae3dModel } from './vae3d/modelIR'

type Row = readonly [id: string, type: ModelNodeType, en: string, zh: string, enDescription: string, zhDescription: string, shape?: TensorShape, enNote?: string, zhNote?: string]
type Path = readonly string[]
type ExtraEdge = readonly [source: string, target: string, type?: ModelEdge['type']]
export type StudyCategory = 'pi' | 'diffusion' | 'video'
export interface StudyEntry { model: ModelIR; category: StudyCategory; kind: 'architecture' | 'method' | 'overview'; reference?: string }

function study(
  id: string, name: string, zhName: string, description: string, zhDescription: string,
  category: StudyCategory, kind: StudyEntry['kind'], rows: Row[], paths: Path[], extras: ExtraEdge[] = [], reference?: string,
): StudyEntry {
  const edges: ModelEdge[] = []
  const seen = new Set<string>()
  function add(source: string, target: string, type: ModelEdge['type'] = 'data') {
    const key = source + '>' + target
    if (seen.has(key)) return
    seen.add(key)
    edges.push({ id: key, source, target, type })
  }
  paths.forEach((path) => path.slice(1).forEach((target, index) => add(path[index], target)))
  extras.forEach(([source, target, type]) => add(source, target, type))
  const nodes: ModelNode[] = rows.map(([nodeId, type, en, zh, enDescription, zhDescription, outputShape, enNote, zhNote]) => {
    const incoming = edges.find((edge) => edge.target === nodeId)
    const inputShape = rows.find((row) => row[0] === incoming?.source)?.[6]
    return {
      id: nodeId, type, name: en, description: enDescription, inputShape, outputShape,
      learning: { what: enDescription, notes: enNote },
      localized: { zh: { name: zh, description: zhDescription, learning: { what: zhDescription, notes: zhNote } } },
    }
  })
  return { category, kind, reference, model: withDefaultLayout({
    id, name, description, localized: { zh: { name: zhName, description: zhDescription } }, nodes, edges,
  }) }
}

const pi0 = study('pi0', 'π0', 'π0', 'Vision-language-action flow policy with a PaliGemma prefix and an action expert.', '以 PaliGemma 作为视觉语言前缀、以动作专家预测连续动作流的 VLA 策略。', 'pi', 'architecture', [
  ['image', 'tensor', 'Camera images', '相机图像', 'Visual observations from one or more cameras.', '一个或多个相机提供的视觉观测。', ['B', 3, 'H', 'W']],
  ['instruction', 'tensor', 'Instruction', '语言指令', 'The task is expressed in natural language.', '自然语言说明机器人要完成的任务。', ['B', 'L']],
  ['state', 'tensor', 'Robot state', '机器人状态', 'Proprioceptive state conditions action generation.', '本体状态为动作生成提供当前关节或末端信息。', ['B', 'Dstate']],
  ['vlm', 'encoder', 'PaliGemma prefix', 'PaliGemma 前缀', 'The pretrained VLM encodes image and language context.', '预训练视觉语言模型编码图像和指令上下文。', ['B', 'Nprefix', 'D'], 'The prefix does not need to read future action tokens.', '前缀无需读取未来动作 token。'],
  ['noisy', 'tensor', 'Noisy action chunk', '带噪动作块', 'A noisy continuous action sequence is the flow state.', '一段带噪连续动作序列构成流模型的当前状态。', ['B', 'Horizon', 'Daction']],
  ['time', 'tensor', 'Flow time τ', '流时间 τ', 'The action expert is conditioned on the current flow time.', '动作专家以当前流时间为条件。', ['B']],
  ['expert', 'attention', 'Action expert', '动作专家', 'A dedicated transformer reads the prefix, robot state, flow time and noisy actions.', '独立动作专家读取前缀、机器人状态、流时间和带噪动作。', ['B', 'Horizon', 'D']],
  ['velocity', 'linear', 'Velocity head', '速度预测头', 'Predicts the vector field for the action flow.', '预测动作流的速度场。', ['B', 'Horizon', 'Daction']],
  ['integrate', 'generic', 'ODE integration', 'ODE 积分', 'Repeated updates move the noisy actions toward a usable action chunk.', '多次更新把带噪动作推进到可执行动作块。', ['B', 'Horizon', 'Daction']],
  ['actions', 'tensor', 'Action chunk', '动作块', 'The policy outputs several future actions together.', '策略一次输出未来一段动作，而非逐 token 自回归。', ['B', 'Horizon', 'Daction']],
], [['image', 'vlm', 'expert', 'velocity', 'integrate', 'actions']], [['instruction', 'vlm', 'condition'], ['state', 'expert', 'condition'], ['noisy', 'expert', 'condition'], ['time', 'expert', 'condition']], 'https://arxiv.org/abs/2410.24164')

const pi05 = study('pi05', 'π0.5', 'π0.5', 'A π0-based policy co-trained across robot, web and semantic tasks for open-world generalization.', '基于 π0，通过机器人、网页与语义任务协同训练来提升开放环境泛化。', 'pi', 'architecture', [
  ['image', 'tensor', 'Visual observation', '视觉观测', 'Camera views show the current scene.', '相机视角呈现当前场景。', ['B', 3, 'H', 'W']],
  ['instruction', 'tensor', 'Task command', '任务指令', 'A high-level request can describe an entire task.', '高层指令可以描述完整任务。', ['B', 'L']],
  ['vlm', 'encoder', 'Shared VLM', '共享 VLM', 'A vision-language backbone connects perception with semantic reasoning.', '视觉语言骨干将感知与语义推理连接起来。', ['B', 'N', 'D']],
  ['subtask', 'generic', 'Semantic subtask', '语义子任务', 'High-level prediction selects the next meaningful step.', '高层预测选出下一步有意义的子任务。', ['B', 'Lsub']],
  ['state', 'tensor', 'Robot state', '机器人状态', 'Proprioception enters the multimodal observation prefix.', '本体状态进入多模态观测前缀。', ['B', 'Dstate']],
  ['expert', 'attention', 'Low-level action expert', '低层动作专家', 'Added during post-training, the action expert predicts the flow vector field conditioned on context.', '动作专家在后训练阶段加入，根据上下文预测动作流的速度场。', ['B', 'Horizon', 'D']],
  ['noisy', 'tensor', 'Noisy action chunk', '带噪动作块', 'The flow state is a noisy continuous action chunk.', '流状态是带噪的连续动作块。', ['B', 'Horizon', 'Daction']],
  ['time', 'tensor', 'Flow time τ', '流时间 τ', 'The action expert reads the current flow time.', '动作专家读取当前流时间。', ['B']],
  ['solver', 'generic', 'Flow integration', '流积分', 'Integrating the predicted vector field produces actions.', '积分预测速度场后得到动作。', ['B', 'Horizon', 'Daction']],
  ['actions', 'tensor', 'Action chunk', '动作块', 'A short sequence of controls is executed before replanning.', '执行短时控制序列后再根据新观测规划。', ['B', 'Horizon', 'Daction']],
  ['cotrain', 'generic', 'Heterogeneous pre-training', '异构数据预训练', 'Discrete-token robot, semantic and web tasks first adapt the VLM.', '机器人离散动作、语义与网页任务首先用于适配 VLM。', undefined, 'The flow action expert is added in post-training; this node is not an inference block.', '流动作专家在后训练阶段加入；此节点不是推理模块。'],
], [['image', 'vlm', 'subtask', 'expert', 'solver', 'actions']], [['instruction', 'vlm', 'condition'], ['state', 'vlm', 'condition'], ['vlm', 'expert', 'condition'], ['noisy', 'expert', 'condition'], ['time', 'expert', 'condition'], ['cotrain', 'vlm', 'training']], 'https://arxiv.org/abs/2504.16054')

const piSeries = study('pi-series', 'π series map', 'π 系列脉络', 'A learning map of the π policy family, not a single network graph.', 'π 系列策略的学习脉络图，不代表一张单一网络计算图。', 'pi', 'overview', [
  ['pi0', 'generic', 'π0', 'π0', 'A VLM prefix plus a flow-based action expert predicts continuous action chunks.', 'VLM 前缀与 Flow 动作专家共同生成连续动作块。'],
  ['pi05', 'generic', 'π0.5', 'π0.5', 'Adds heterogeneous co-training and semantic task structure for generalization.', '引入异构协同训练与语义任务结构以提升泛化。'],
  ['pi07', 'generic', 'π0.7', 'π0.7', 'The later family member emphasizes steerability and composing known skills for new tasks.', '后续版本强调可引导性与组合已有技能完成新任务。', undefined, 'This is a capability summary, not an undisclosed layer-by-layer architecture.', '这里只总结公开能力，不虚构未公开的逐层结构。'],
  ['vision', 'tensor', 'Vision + language', '视觉与语言', 'Shared multimodal context remains central to the family.', '多模态上下文仍是这一系列的共同核心。'],
  ['action', 'tensor', 'Robot actions', '机器人动作', 'All variants ultimately turn context into robot control.', '各版本最终都将上下文转化为机器人控制。'],
], [['pi0', 'pi05', 'pi07']], [['vision', 'pi0', 'condition'], ['pi07', 'action', 'branch']], 'https://www.pi.website/blog/pi07')

const ddpm = study('ddpm', 'DDPM', 'DDPM', 'Denoising diffusion: train a noise predictor, then reverse a gradual noise process.', '去噪扩散：训练噪声预测器，再逐步逆转加噪过程。', 'diffusion', 'architecture', [
  ['clean', 'tensor', 'Clean sample x₀', '干净样本 x₀', 'A training image sampled from the data distribution.', '从数据分布取出的训练图像。', ['B', 3, 'H', 'W']],
  ['t', 'tensor', 'Time step t', '时间步 t', 'The noise schedule determines the selected noise level.', '噪声日程确定当前噪声强度。', ['B']],
  ['epsilon', 'tensor', 'Gaussian noise ε', '高斯噪声 ε', 'Independent noise is sampled for the training example.', '为训练样本采样独立高斯噪声。', ['B', 3, 'H', 'W']],
  ['forward', 'generic', 'Forward noising', '前向加噪', 'Constructs xₜ directly from x₀ and ε.', '直接由 x₀ 和 ε 构造 xₜ。', ['B', 3, 'H', 'W'], 'xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε', 'xₜ = √ᾱₜ x₀ + √(1−ᾱₜ) ε'],
  ['denoiser', 'encoder', 'Noise predictor', '噪声预测器', 'A time-conditioned network estimates the added noise.', '带时间条件的网络估计被加入的噪声。', ['B', 3, 'H', 'W']],
  ['loss', 'generic', 'Noise MSE', '噪声 MSE', 'Training compares predicted noise with the sampled ε.', '训练时比较预测噪声与采样的 ε。', ['B']],
  ['prior', 'tensor', 'Sampling prior xT', '采样先验 xT', 'Inference starts from fresh Gaussian noise.', '推理从新的高斯噪声开始。', ['B', 3, 'H', 'W']],
  ['reverse', 'generic', 'Reverse steps', '反向采样步骤', 'Starting from Gaussian noise, repeated steps produce a sample.', '从高斯噪声开始，反复更新得到样本。', ['B', 3, 'H', 'W']],
  ['sample', 'tensor', 'Generated sample', '生成样本', 'The final low-noise state is the generated image.', '最终低噪状态即生成图像。', ['B', 3, 'H', 'W']],
], [['clean', 'forward', 'denoiser', 'loss'], ['prior', 'reverse', 'sample']], [['epsilon', 'forward', 'training'], ['epsilon', 'loss', 'training'], ['t', 'forward', 'condition'], ['t', 'denoiser', 'condition'], ['denoiser', 'reverse', 'reuse']], 'https://arxiv.org/abs/2006.11239')

const ddim = study('ddim', 'DDIM sampler', 'DDIM 采样器', 'A faster sampling path that can reuse a DDPM-trained noise predictor.', '可复用 DDPM 噪声预测器的跳步采样路径。', 'diffusion', 'method', [
  ['noise', 'tensor', 'Initial noise xₜ', '初始噪声 xₜ', 'Start from a high noise level.', '从较高噪声等级开始。', ['B', 3, 'H', 'W']],
  ['schedule', 'generic', 'Selected time grid', '选定时间网格', 'A sparse sequence of time steps skips many training indices.', '稀疏时间步序列跳过许多训练索引。'],
  ['predictor', 'encoder', 'Noise predictor', '噪声预测器', 'The existing model predicts ε at the current time.', '已有模型预测当前时间步的 ε。', ['B', 3, 'H', 'W']],
  ['x0', 'generic', 'Estimate clean x₀', '估计干净 x₀', 'Combines xₜ and predicted noise to estimate the clean sample.', '结合 xₜ 与预测噪声估计干净样本。', ['B', 3, 'H', 'W']],
  ['jump', 'generic', 'DDIM update t→s', 'DDIM 跳步 t→s', 'Projects toward an earlier chosen noise level.', '投影到选定的更早噪声等级。', ['B', 3, 'H', 'W']],
  ['eta', 'generic', 'Stochasticity η', '随机性 η', 'η=0 gives the deterministic DDIM path.', 'η=0 对应确定性 DDIM 路径。'],
  ['sample', 'tensor', 'Final sample', '最终样本', 'Repeat the update across the selected time grid.', '沿选定时间网格重复更新。', ['B', 3, 'H', 'W']],
], [['noise', 'predictor', 'x0', 'jump', 'sample']], [['noise', 'x0', 'branch'], ['noise', 'jump', 'branch'], ['schedule', 'predictor', 'condition'], ['schedule', 'jump', 'condition'], ['eta', 'jump', 'condition'], ['jump', 'predictor', 'feedback']], 'https://arxiv.org/abs/2010.02502')

const cfg = study('cfg', 'Classifier-free guidance', '无分类器引导 CFG', 'A guidance module that combines conditional and unconditional predictions.', '组合有条件与无条件预测的引导模块，不是一套独立的去噪网络。', 'diffusion', 'method', [
  ['xt', 'tensor', 'Noisy state xₜ', '带噪状态 xₜ', 'The same noisy state is evaluated twice.', '同一个带噪状态分别进入两次预测。', ['B', 3, 'H', 'W']],
  ['condition', 'tensor', 'Condition c', '条件 c', 'A text prompt or class label provides the desired direction.', '文本提示或类别标签提供目标方向。', ['B', 'L']],
  ['uncond', 'attention', 'Unconditional pass', '无条件预测', 'The denoiser sees an empty condition.', '去噪器读取空条件。', ['B', 3, 'H', 'W']],
  ['cond', 'attention', 'Conditional pass', '有条件预测', 'The same denoiser sees the supplied condition.', '同一个去噪器读取给定条件。', ['B', 3, 'H', 'W']],
  ['combine', 'generic', 'Guidance blend', '引导组合', 'ε̂ = εuncond + w(εcond − εuncond).', 'ε̂ = ε无条件 + w(ε有条件 − ε无条件)。', ['B', 3, 'H', 'W'], 'Guidance scale w changes the conditioning strength.', '引导系数 w 改变条件约束强度。'],
  ['solver', 'generic', 'Sampler step', '采样器更新', 'The combined estimate is passed to the sampler.', '组合后的估计交给采样器更新。', ['B', 3, 'H', 'W']],
], [['xt', 'uncond', 'combine', 'solver'], ['xt', 'cond', 'combine']], [['condition', 'cond', 'condition']], 'https://arxiv.org/abs/2207.12598')

const ldm = study('latent-diffusion', 'Latent Diffusion + VAE', '潜空间扩散 + VAE', 'A pretrained autoencoder compresses images before diffusion and decodes them afterward.', '预训练自编码器先压缩图像，扩散在潜空间进行，最后再解码。', 'diffusion', 'architecture', [
  ['image', 'tensor', 'Training image', '训练图像', 'The pixel-space image enters the pretrained autoencoder.', '像素空间图像输入预训练自编码器。', ['B', 3, 256, 256]],
  ['encode', 'encoder', 'VAE encoder', 'VAE 编码器', 'Compresses spatial dimensions while retaining useful image structure.', '压缩空间维度并保留重要图像结构。', ['B', 4, 32, 32]],
  ['latent', 'latent', 'Image latent z', '图像潜变量 z', 'Diffusion operates on this smaller feature grid.', '扩散过程在更小的特征网格上执行。', ['B', 4, 32, 32]],
  ['noise', 'generic', 'Latent noising', '潜变量加噪', 'A noisy latent zₜ is constructed for training.', '训练时构造带噪潜变量 zₜ。', ['B', 4, 32, 32]],
  ['text', 'tensor', 'Text condition', '文本条件', 'Text embeddings guide generation through cross-attention.', '文本嵌入通过交叉注意力引导生成。', ['B', 'L', 'D']],
  ['unet', 'encoder', 'Conditional U-Net', '条件 U-Net', 'Predicts noise in latent space, often with cross-attention.', '在潜空间预测噪声，通常含交叉注意力。', ['B', 4, 32, 32]],
  ['prior', 'tensor', 'Gaussian latent prior', '高斯潜变量先验', 'Generation begins from fresh latent-shaped noise.', '生成从新的潜变量形状高斯噪声开始。', ['B', 4, 32, 32]],
  ['sample', 'generic', 'Latent sampler', '潜空间采样器', 'Iterative updates produce a clean latent.', '迭代更新得到干净潜变量。', ['B', 4, 32, 32]],
  ['decode', 'decoder', 'VAE decoder', 'VAE 解码器', 'Maps the generated latent back to pixels.', '将生成潜变量还原到像素空间。', ['B', 3, 256, 256]],
], [['image', 'encode', 'latent', 'noise', 'unet'], ['prior', 'sample', 'decode']], [['text', 'unet', 'condition'], ['unet', 'sample', 'reuse']], 'https://arxiv.org/abs/2112.10752')

const unet = study('unet', 'Diffusion U-Net', '扩散 U-Net', 'A multiscale denoiser with an encoder, bottleneck, decoder and skip connections.', '具有编码路径、瓶颈、解码路径和跳跃连接的多尺度去噪器。', 'diffusion', 'architecture', [
  ['input', 'tensor', 'Noisy input xₜ', '带噪输入 xₜ', 'The noisy image or latent enters the network.', '带噪图像或潜变量进入网络。', ['B', 4, 64, 64]],
  ['time', 'tensor', 'Time embedding', '时间嵌入', 'The noise level conditions residual blocks.', '噪声等级作为残差块的条件。', ['B', 'D']],
  ['down1', 'downsample', 'Down block 1', '下采样块 1', 'Local features are extracted at high resolution.', '在高分辨率上提取局部特征。', ['B', 64, 64, 64]],
  ['down2', 'downsample', 'Down block 2', '下采样块 2', 'Features gain a larger receptive field as resolution falls.', '分辨率降低时感受野扩大。', ['B', 128, 32, 32]],
  ['mid', 'attention', 'Bottleneck', '瓶颈层', 'Deep features mix information at low resolution.', '深层特征在低分辨率下混合信息。', ['B', 128, 32, 32]],
  ['up2', 'upsample', 'Up block 2', '上采样块 2', 'Decoder features fuse with corresponding encoder features.', '解码特征与相应编码特征融合。', ['B', 64, 64, 64]],
  ['up1', 'upsample', 'Up block 1', '上采样块 1', 'The representation returns to input resolution.', '表示恢复到输入分辨率。', ['B', 64, 64, 64]],
  ['output', 'tensor', 'Noise estimate', '噪声估计', 'A final projection predicts noise or another diffusion target.', '最终投影预测噪声或其他扩散目标。', ['B', 4, 64, 64]],
], [['input', 'down1', 'down2', 'mid', 'up2', 'up1', 'output']], [['time', 'down1', 'condition'], ['time', 'down2', 'condition'], ['time', 'mid', 'condition'], ['time', 'up2', 'condition'], ['time', 'up1', 'condition'], ['down2', 'up2', 'branch'], ['down1', 'up1', 'branch']], 'https://arxiv.org/abs/2006.11239')

const dit = study('dit', 'DiT', 'DiT', 'A diffusion Transformer that replaces a U-Net with Transformer blocks over latent patches.', '在潜变量 patch 上用 Transformer 块替代 U-Net 的扩散骨干。', 'diffusion', 'architecture', [
  ['latent', 'latent', 'Noisy latent', '带噪潜变量', 'The VAE latent is the diffusion state.', 'VAE 潜变量构成扩散状态。', ['B', 4, 32, 32]],
  ['patch', 'generic', 'Patch embedding', 'Patch 嵌入', 'Spatial patches become a token sequence.', '空间 patch 转为 token 序列。', ['B', 256, 'D']],
  ['position', 'tensor', 'Position encoding', '位置编码', 'Position information distinguishes patches.', '位置信息用于区分各个 patch。', ['B', 256, 'D']],
  ['condition', 'tensor', 'Time + class', '时间 + 类别', 'The denoising step and optional label form conditioning c.', '去噪时间步和可选类别形成条件 c。', ['B', 'D']],
  ['adaln', 'norm', 'AdaLN-Zero ①', 'AdaLN-Zero ①', 'Condition-dependent shift and scale modulate the attention input.', '条件产生的平移与缩放调制注意力输入。', ['B', 256, 'D']],
  ['attention', 'attention', 'Self-attention', '自注意力', 'The first gated residual branch mixes patch tokens.', '第一条门控残差支路混合 patch token。', ['B', 256, 'D']],
  ['add1', 'generic', 'Residual add ①', '残差相加 ①', 'Add the gated attention output to the incoming tokens.', '门控注意力输出与输入 token 相加。', ['B', 256, 'D']],
  ['adaln2', 'norm', 'AdaLN-Zero ②', 'AdaLN-Zero ②', 'A second condition-dependent modulation precedes the MLP.', '第二次条件调制位于 MLP 前。', ['B', 256, 'D']],
  ['mlp', 'mlp', 'MLP', 'MLP', 'The second gated branch refines each token.', '第二条门控支路更新每个 token。', ['B', 256, 'D']],
  ['add2', 'generic', 'Residual add ②', '残差相加 ②', 'Add the gated MLP output to the attention residual stream.', '门控 MLP 输出与注意力残差流相加。', ['B', 256, 'D']],
  ['head', 'linear', 'Final projection', '最终投影', 'Tokens are projected and unpatchified to a latent prediction.', 'token 经投影与反 patch 化得到潜变量预测。', ['B', 4, 32, 32]],
], [['latent', 'patch', 'position', 'adaln', 'attention', 'add1', 'adaln2', 'mlp', 'add2', 'head']], [['condition', 'adaln', 'condition'], ['condition', 'adaln2', 'condition'], ['position', 'add1', 'residual'], ['add1', 'add2', 'residual']], 'https://arxiv.org/abs/2212.09748')

const flow = study('flow-matching', 'Flow Matching', 'Flow Matching', 'A training and sampling framework that learns a velocity field between noise and data.', '在噪声与数据之间学习速度场的训练和采样框架。', 'diffusion', 'method', [
  ['noise', 'tensor', 'Noise x₀', '噪声 x₀', 'One endpoint is sampled from a simple prior.', '起点从简单先验分布采样。', ['B', 'D']],
  ['data', 'tensor', 'Data x₁', '数据 x₁', 'The other endpoint is a real training sample.', '终点是真实训练样本。', ['B', 'D']],
  ['time', 'tensor', 'Flow time t', '流时间 t', 'A continuous time value selects a point on the path.', '连续时间值选取路径上的一点。', ['B']],
  ['interpolate', 'generic', 'Probability path', '概率路径', 'Constructs xₜ from paired noise and data.', '由配对噪声与数据构造 xₜ。', ['B', 'D'], 'For a linear path, xₜ = (1−t)x₀ + tx₁.', '线性路径下 xₜ = (1−t)x₀ + tx₁。'],
  ['target', 'tensor', 'Target velocity', '目标速度', 'For a linear path the conditional target is x₁ − x₀.', '线性路径的条件目标速度是 x₁ − x₀。', ['B', 'D']],
  ['network', 'attention', 'Velocity model', '速度网络', 'The model predicts a vector field from xₜ and t.', '网络根据 xₜ 与 t 预测速度场。', ['B', 'D']],
  ['loss', 'generic', 'Velocity loss', '速度损失', 'Regression compares predicted and target velocities.', '回归损失比较预测速度与目标速度。', ['B']],
  ['prior', 'tensor', 'Fresh noise', '新采样噪声', 'Inference draws a new prior sample.', '推理阶段重新采样一个先验样本。', ['B', 'D']],
  ['ode', 'generic', 'ODE sampler', 'ODE 采样器', 'Numerical integration transports new noise toward data.', '数值积分将新噪声逐步推向数据。', ['B', 'D']],
], [['noise', 'interpolate', 'network', 'loss'], ['prior', 'ode']], [['data', 'interpolate', 'training'], ['time', 'interpolate', 'condition'], ['time', 'network', 'condition'], ['noise', 'target', 'training'], ['data', 'target', 'training'], ['target', 'loss', 'training'], ['network', 'ode', 'reuse']], 'https://arxiv.org/abs/2210.02747')

const video = study('video-generation', 'Video generation pipeline', '视频生成全流程', 'A learning map from video compression through spatiotemporal generation to decoded frames.', '从视频压缩、时空建模到帧解码的完整学习图谱。', 'video', 'overview', [
  ['frames', 'tensor', 'Video frames', '视频帧', 'A clip has batch, channel, time, height and width axes.', '视频片段包含批次、通道、时间、高和宽。', ['B', 3, 129, 256, 256]],
  ['vae', 'encoder', '3D VAE / AE', '3D VAE / AE', 'A spatiotemporal autoencoder compresses frames into video latents.', '时空自编码器将视频帧压缩到视频潜空间。', ['B', 16, 'T′', 'H′', 'W′']],
  ['latent', 'latent', 'Training video latent', '训练视频潜变量', 'Encoded clean videos provide the training target and latent statistics.', '编码后的干净视频用于训练目标与潜变量分布。', ['B', 16, 'T′', 'H′', 'W′'], 'This is the training branch; inference starts from noise.', '这是训练分支；生成推理从噪声开始。'],
  ['noise', 'tensor', 'Noisy video latent', '带噪视频潜变量', 'Training adds noise to encoded video; inference initializes this state from Gaussian noise.', '训练时对编码视频加噪；推理时由高斯噪声初始化该状态。', ['B', 16, 'T′', 'H′', 'W′']],
  ['noising', 'generic', 'Training noising', '训练加噪', 'Construct a noisy latent and target from the clean video latent.', '由干净视频潜变量构造带噪潜变量和训练目标。', ['B', 16, 'T′', 'H′', 'W′']],
  ['prior', 'tensor', 'Inference noise', '推理起始噪声', 'Sampling starts from a fresh Gaussian video latent.', '采样从新的高斯视频潜变量开始。', ['B', 16, 'T′', 'H′', 'W′']],
  ['patch', 'generic', 'Space-time patches', '时空 Patch', 'The noisy latent volume is split into space-time tokens.', '带噪潜变量体被划分成时空 token。', ['B', 'N', 'D']],
  ['prompt', 'tensor', 'Text prompt', '文本提示', 'A prompt supplies semantic conditioning.', '提示词提供语义条件。', ['B', 'L']],
  ['text', 'encoder', 'Text encoders', '文本编码器', 'Text embeddings become conditioning tokens.', '文本嵌入转为条件 token。', ['B', 'L', 'Dtext']],
  ['time', 'tensor', 'Noise time t', '噪声时间 t', 'The diffusion or flow time conditions the denoising backbone.', '扩散或流时间作为去噪骨干的条件。', ['B']],
  ['temporal', 'attention', 'Space-time DiT', '时空 DiT', 'Spatial and temporal attention model the noisy video tokens.', '空间和时间注意力建模带噪视频 token。', ['B', 'N', 'D'], 'Open-Sora STDiT separates spatial and temporal attention; causal masking is a separate choice.', 'Open-Sora STDiT 分开处理空间与时间注意力；因果掩码是另一个独立选择。'],
  ['solver', 'generic', 'Sampling solver', '采样求解器', 'Repeated denoising or flow updates yield a clean video latent.', '多次去噪或流更新得到干净视频潜变量。', ['B', 16, 'T′', 'H′', 'W′']],
  ['decode', 'decoder', 'Video decoder', '视频解码器', 'The decoder reconstructs frames from the generated latent.', '解码器从生成潜变量重建视频帧。', ['B', 3, 129, 256, 256]],
], [['frames', 'vae', 'latent', 'noising', 'noise', 'patch', 'temporal'], ['prior', 'solver', 'decode'], ['prompt', 'text']], [['text', 'temporal', 'condition'], ['time', 'temporal', 'condition'], ['temporal', 'solver', 'reuse']], 'https://arxiv.org/abs/2412.20404')

const causalVideo = study('causal-video', 'Causal video generation', '因果视频生成', 'A blockwise continuation map: later video blocks may use past context but cannot see future blocks.', '按块续写视频：后续块可读取历史上下文，但不能看到未来块。', 'video', 'overview', [
  ['prefix', 'tensor', 'Observed prefix', '已观测前缀', 'Available frames form the causal context.', '已有视频帧构成因果上下文。', ['B', 3, 'Tpast', 'H', 'W']],
  ['encode', 'encoder', 'Causal 3D VAE', '因果 3D VAE', 'The encoder must avoid peeking at future frames.', '编码器不得偷看未来帧。', ['B', 'Cz', 'T′past', 'H′', 'W′']],
  ['cache', 'generic', 'Past latent cache', '历史潜变量缓存', 'Previously generated clean latents are retained as context.', '此前生成的干净潜变量作为上下文保存。', ['B', 'Npast', 'D']],
  ['mask', 'generic', 'Block-causal mask', '分块因果掩码', 'A block may attend to earlier blocks, never to future blocks.', '当前块可关注历史块，不能关注未来块。', ['N', 'N']],
  ['noisy', 'tensor', 'Current noisy block', '当前带噪块', 'Only the next block is initialized with fresh noise.', '仅下一块由新噪声初始化。', ['B', 'Nblock', 'D']],
  ['attention', 'attention', 'Causal space-time model', '因果时空模型', 'Combines current noisy tokens with allowed past context.', '将当前带噪 token 与允许的历史上下文结合。', ['B', 'Nblock', 'D']],
  ['sample', 'generic', 'Denoise next block', '生成下一视频块', 'Sampling resolves the current block before moving forward.', '先完成当前块的采样，再向后推进。', ['B', 'Nblock', 'D']],
  ['append', 'generic', 'Append to context', '追加到上下文', 'The clean block becomes available to the following block.', '干净视频块随后成为下一块可用的历史。', ['B', 'Npast+Nblock', 'D']],
  ['decode', 'decoder', 'Decode frames', '解码视频帧', 'The video decoder reconstructs the generated block.', '视频解码器重建生成的视频块。', ['B', 3, 'Tblock', 'H', 'W']],
], [['prefix', 'encode', 'cache', 'attention', 'sample', 'append'], ['sample', 'decode']], [['noisy', 'attention', 'condition'], ['mask', 'attention', 'condition'], ['append', 'cache', 'feedback']], 'https://arxiv.org/abs/2412.07772')

// Paper diagrams use stable lanes. Coordinates are grid units, not model dimensions.
// Main inference/data flow runs horizontally; inputs and conditions occupy adjacent lanes.
const diagrams: Record<string, Record<string, [number, number]>> = {
  vae3d: { input: [0, 0], conv3d: [1, 0], downsample: [2, 0], mid: [3, 0], encoder: [3, -1], moments: [2, -1], mu: [1, -0.55], logvar: [1, -1.45], sampling: [0, -1], latent: [0, -2.45], decoder: [1, -2.45], output: [2, -2.45] },
  pi0: { image: [0, 0], instruction: [0, 1], vlm: [1, 0], state: [1, -1], noisy: [2, -1], time: [3, -1], expert: [2, 0], velocity: [3, 0], integrate: [4, 0], actions: [5, 0] },
  pi05: { image: [0, 0], instruction: [0, 1], state: [1, 1], vlm: [1, 0], subtask: [2, 0], expert: [3, 0], solver: [4, 0], actions: [5, 0], noisy: [3, -1], time: [4, -1], cotrain: [1, -1.5] },
  'pi-series': { vision: [0, 1], pi0: [0, 0], pi05: [1, 0], pi07: [2, 0], action: [2, -1] },
  ddpm: { clean: [0, 1], epsilon: [0, 2], t: [1, 2], forward: [1, 1], denoiser: [2, 1], loss: [3, 1], prior: [0, -1], reverse: [2, -1], sample: [3, -1] },
  ddim: { noise: [0, 0], predictor: [1, 0], x0: [2, 0], jump: [3, 0], sample: [4, 0], schedule: [1.5, 1], eta: [3, 1] },
  cfg: { xt: [0, 0], uncond: [1, 0.65], cond: [1, -0.65], combine: [2, 0], solver: [3, 0], condition: [0, -1.3] },
  'latent-diffusion': { image: [0, 1], encode: [1, 1], latent: [2, 1], noise: [3, 1], unet: [4, 1], text: [4, 2], prior: [2, -1], sample: [4, -1], decode: [5, -1] },
  unet: { input: [0, 2], down1: [1, 2], down2: [2, 1], mid: [3, 0], up2: [4, 1], up1: [5, 2], output: [6, 2], time: [3, 3.5] },
  dit: { latent: [0, 1], patch: [1, 1], position: [2, 1], adaln: [3, 1], attention: [4, 1], add1: [5, 1], adaln2: [5, -1], mlp: [4, -1], add2: [3, -1], head: [2, -1], condition: [3, 2.3] },
  'flow-matching': { noise: [0, 1], data: [0, 2], time: [1, 2], interpolate: [1, 1], target: [2, 2], network: [2, 1], loss: [3, 1], prior: [0, -1], ode: [2, -1] },
  'video-generation': { frames: [0, 1], vae: [1, 1], latent: [2, 1], noising: [3, 1], noise: [4, 1], patch: [4, -1], temporal: [3, -1], prior: [4, -2], time: [3, -2], solver: [2, -1], decode: [1, -1], prompt: [5, 1], text: [5, 0] },
  'causal-video': { prefix: [0, 0], encode: [1, 0], cache: [2, 0], attention: [3, 0], sample: [4, 0], append: [5, 0], decode: [4, -1], mask: [3, 1], noisy: [3, -1] },
}

function arrange(entry: StudyEntry): StudyEntry {
  const grid = diagrams[entry.model.id]
  const nodes = entry.model.nodes.map((node) => {
    const cell = grid[node.id]
    if (!cell) throw new Error(`Missing diagram position: ${entry.model.id}/${node.id}`)
    return { ...node, visualization: { ...node.visualization, position: [cell[0] * 3.35, cell[1] * 3.1, 0] as [number, number, number] } }
  })
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const edges = entry.model.edges.map((edge) => {
    const pos = (id: string) => byId.get(id)!.visualization.position!
    if (entry.model.id === 'ddim' && edge.source === 'noise' && (edge.target === 'x0' || edge.target === 'jump')) {
      const a = pos(edge.source)
      const b = pos(edge.target)
      const rail = edge.target === 'x0' ? -2.25 : -3.1
      return { ...edge, via: [[a[0], rail, -0.65], [b[0], rail, -0.65]] as [number, number, number][] }
    }
    if (entry.model.id === 'ddpm' && edge.source === 'epsilon' && edge.target === 'loss') {
      const a = pos(edge.source)
      const b = pos(edge.target)
      const rail = a[1] + 1.8
      return { ...edge, via: [[a[0], rail, -0.65], [b[0], rail, -0.65]] as [number, number, number][] }
    }
    if (entry.model.id === 'pi05' && edge.source === 'vlm' && edge.target === 'expert') {
      const a = pos(edge.source)
      const b = pos(edge.target)
      const rail = 4.75
      return { ...edge, via: [[a[0] + 1.15, a[1], -0.7], [a[0] + 1.15, rail, -0.7], [b[0], rail, -0.7]] as [number, number, number][] }
    }
    if (entry.model.id === 'dit' && edge.source === 'condition' && edge.target === 'adaln2') {
      const a = byId.get(edge.source)!.visualization.position!
      const b = byId.get(edge.target)!.visualization.position!
      return { ...edge, via: [[a[0] + 7.5, a[1], -0.65], [a[0] + 7.5, b[1], -0.65]] as [number, number, number][] }
    }
    if (edge.type !== 'residual' && edge.type !== 'feedback') return edge
    const a = byId.get(edge.source)!.visualization.position!
    const b = byId.get(edge.target)!.visualization.position!
    const rail = edge.type === 'feedback' ? Math.max(a[1], b[1]) + 5.15
      : entry.model.id === 'dit' && edge.source === 'add1' ? Math.min(a[1], b[1]) - 1.75
      : Math.max(a[1], b[1]) + 1.75
    return { ...edge, via: [[a[0], rail, -0.65], [b[0], rail, -0.65]] as [number, number, number][] }
  })
  return { ...entry, model: { ...entry.model, nodes, edges } }
}

export const studyCatalog: StudyEntry[] = ([
  { model: vae3dModel, category: 'video', kind: 'architecture' },
  pi0, pi05, piSeries, ddpm, ddim, cfg, ldm, dit, unet, flow, video, causalVideo,
] as StudyEntry[]).map(arrange)

export const studyById = Object.fromEntries(studyCatalog.map((entry) => [entry.model.id, entry])) as Record<string, StudyEntry>

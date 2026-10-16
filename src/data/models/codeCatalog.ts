// Built-in models defined by editable PyTorch source code.
import mlp from './code/mlp.py?raw'
import lenet from './code/lenet.py?raw'
import resnet from './code/resnet.py?raw'
import gpt from './code/gpt.py?raw'
import vit from './code/vit.py?raw'
import unet from './code/unet.py?raw'
import dit from './code/dit.py?raw'
import vae from './code/vae.py?raw'
import lstm from './code/lstm.py?raw'
import ddpm from './code/ddpm.py?raw'
import latentDiffusion from './code/latent_diffusion.py?raw'
import ddim from './code/ddim.py?raw'
import cfg from './code/cfg.py?raw'
import flowMatching from './code/flow_matching.py?raw'
import pi0 from './code/pi0.py?raw'
import pi05 from './code/pi05.py?raw'
import videoDit from './code/video_dit.py?raw'
import causalVideo from './code/causal_video.py?raw'
import gptTrain from './code/gpt_train.py?raw'
import uniadet from './code/uniadet.py?raw'
import visualad from './code/visualad.py?raw'
import templateConfig from '../../../templates/atlas-template/atlas.json?raw'
import templateModel from '../../../templates/atlas-template/model.py?raw'
import templateData from '../../../templates/atlas-template/data.py?raw'
import templateTrain from '../../../templates/atlas-template/train.py?raw'

export interface Experiment {
  label: { en: string; zh: string }
  hint: { en: string; zh: string }
  find: string
  replace: string
  file?: string
}

export interface CodeModelEntry {
  id: string
  name: string
  zhName: string
  description: string
  zhDescription: string
  family: { en: string; zh: string }
  year?: number
  files: Record<string, string>
  experiments: Experiment[]
  reference?: string
  /** Expansion level to open with (default: chosen automatically). */
  depth?: number
  /** Study map that explains the same system. */
  study?: string
}

const exp = (en: string, zh: string, hintEn: string, hintZh: string, find: string, replace: string): Experiment => ({ label: { en, zh }, hint: { en: hintEn, zh: hintZh }, find, replace })

/** The fixed import template (templates/atlas-template): also the starting point for your own project. */
export const templateFiles: Record<string, string> = { 'model.py': templateModel, 'data.py': templateData, 'train.py': templateTrain, 'atlas.json': templateConfig }

export const codeModels: CodeModelEntry[] = [
  {
    id: 'code-template', name: 'Import template · CNN + training', zhName: '导入模板 · CNN + 训练循环', family: { en: 'Template', zh: '模板' },
    description: 'The fixed import template: model.py, data.py, train.py and atlas.json. One training step is drawn: batch → model → loss → backward → optimizer.',
    zhDescription: '固定导入模板：model.py、data.py、train.py 和 atlas.json。画出一个训练步：批数据 → 模型 → 损失 → 反向传播 → 优化器。',
    files: templateFiles,
    experiments: [
      { ...exp('Only the network', '只看网络', 'With entry model.py the training step disappears and only the forward pass is drawn.', '入口改成 model.py 后训练步消失，只画网络的前向。', '"entry": "train.py"', '"entry": "model.py"'), file: 'atlas.json' },
      { ...exp('Batch size 128', 'batch 改成 128', 'Every tensor in the step gets batch 128; parameters stay the same.', '训练步里所有张量的 batch 维变成 128，参数量不变。', 'BATCH_SIZE = 64', 'BATCH_SIZE = 128'), file: 'data.py' },
      { ...exp('Wider network', '加宽网络', 'widths=(64, 128, 256) changes every ConvBlock.', 'widths=(64, 128, 256) 会改变每个卷积块。', 'widths=(32, 64, 128)', 'widths=(64, 128, 256)'), file: 'model.py' },
    ],
  },
  {
    id: 'code-mlp', name: 'MLP', zhName: '多层感知机 MLP', family: { en: 'Basics', zh: '基础' },
    description: 'Fully connected network for MNIST: the simplest place to watch shapes change.',
    zhDescription: '用于 MNIST 的全连接网络，最适合入门观察形状变化。',
    files: { 'model.py': mlp },
    experiments: [
      exp('Add a hidden layer', '增加一层隐藏层', 'HIDDEN = [1024, 512, 256] adds one Linear → ReLU → Dropout stage.', 'HIDDEN = [1024, 512, 256] 会多出一组 Linear → ReLU → Dropout。', 'HIDDEN = [512, 256]', 'HIDDEN = [1024, 512, 256]'),
      exp('Bigger images', '更大的输入图像', 'A 32×32 input changes the first Linear layer’s in_features.', '32×32 输入会改变第一层 Linear 的 in_features。', 'IMAGE_SIZE = 28', 'IMAGE_SIZE = 32'),
    ],
  },
  {
    id: 'code-lenet', name: 'LeNet-5', zhName: 'LeNet-5', family: { en: 'CNN', zh: '卷积网络' }, year: 1998,
    description: 'The classic convolutional network: convolution, pooling, then fully connected layers.',
    zhDescription: '经典卷积网络：卷积、池化，再接全连接层。',
    files: { 'model.py': lenet },
    experiments: [
      exp('Remove padding (error)', '去掉 padding（会报错）', 'Without padding the feature map ends at 4×4, so Linear(400) no longer matches.', '去掉 padding 后特征图变成 4×4，Linear(400) 不再匹配。', 'padding=2', 'padding=0'),
      exp('Wider conv (error)', '加宽卷积（会报错）', '32 channels × 5 × 5 = 800 features; then fix 16 * 5 * 5 to 32 * 5 * 5.', '32 通道 × 5 × 5 = 800 个特征；再把 16 * 5 * 5 改成 32 * 5 * 5 即可修复。', 'nn.Conv2d(6, 16, kernel_size=5)', 'nn.Conv2d(6, 32, kernel_size=5)'),
      exp('Fix for 32 channels', '修复为 32 通道', 'Matches the classifier input to the wider convolution.', '让分类器输入与加宽后的卷积匹配。', 'nn.Linear(16 * 5 * 5, 120)', 'nn.Linear(32 * 5 * 5, 120)'),
    ],
  },
  {
    id: 'code-resnet', name: 'ResNet-18', zhName: 'ResNet-18 残差网络', family: { en: 'CNN', zh: '卷积网络' }, year: 2015, reference: 'https://arxiv.org/abs/1512.03385',
    description: 'Residual network with BasicBlocks and projection shortcuts. 11.2M parameters.',
    zhDescription: '由 BasicBlock 与投影捷径组成的残差网络，约 1120 万参数。',
    files: { 'model.py': resnet },
    experiments: [
      exp('ResNet-34', '变成 ResNet-34', 'num_blocks=(3, 4, 6, 3) repeats more blocks per stage.', 'num_blocks=(3, 4, 6, 3) 让每个阶段重复更多残差块。', 'model = ResNet(num_blocks=(2, 2, 2, 2))', 'model = ResNet(num_blocks=(3, 4, 6, 3))'),
      exp('Remove the residual', '去掉残差连接', 'The add node and its skip edge disappear from every block.', '每个块里的相加节点和跳跃边都会消失。', 'out = out + self.shortcut(x)        # residual connection · 残差连接', 'out = out  # residual removed · 已去掉残差'),
      exp('64×64 input', '64×64 输入', 'Every feature map doubles in height and width; parameters stay the same.', '每张特征图的高宽都翻倍，参数量不变。', 'x = torch.randn(1, 3, 32, 32)', 'x = torch.randn(1, 3, 64, 64)'),
    ],
  },
  {
    id: 'code-gpt', name: 'GPT (nanoGPT)', zhName: 'GPT 语言模型', family: { en: 'Transformer', zh: 'Transformer' }, year: 2019, reference: 'https://github.com/karpathy/nanoGPT',
    description: 'Decoder-only Transformer with explicit causal attention: QKᵀ, mask, softmax, ·V.',
    zhDescription: '仅解码器 Transformer，显式展示因果注意力：QKᵀ、掩码、softmax、乘 V。',
    files: { 'model.py': gpt },
    experiments: [
      exp('8 layers', '8 层', 'More Transformer blocks appear in the stack.', 'Transformer 块的数量翻倍。', 'GPTConfig(n_layer=4, n_head=4, n_embd=128)', 'GPTConfig(n_layer=8, n_head=4, n_embd=128)'),
      exp('Wider model', '加宽模型', 'n_embd=256 changes every Linear and LayerNorm.', 'n_embd=256 会改变所有 Linear 与 LayerNorm。', 'GPTConfig(n_layer=4, n_head=4, n_embd=128)', 'GPTConfig(n_layer=4, n_head=4, n_embd=256)'),
      exp('5 heads (error)', '5 个注意力头（会报错）', '128 is not divisible by 5, so the assert fails.', '128 不能被 5 整除，assert 会失败。', 'GPTConfig(n_layer=4, n_head=4, n_embd=128)', 'GPTConfig(n_layer=4, n_head=5, n_embd=128)'),
      exp('Longer context', '更长上下文', 'A 64-token input makes the attention maps 64×64.', '64 个 token 的输入让注意力图变为 64×64。', 'torch.randint(0, cfg.vocab_size, (1, 32))', 'torch.randint(0, cfg.vocab_size, (1, 64))'),
    ],
  },
  {
    id: 'code-gpt-train', name: 'GPT · training loop', zhName: 'GPT · 训练循环', family: { en: 'Transformer', zh: 'Transformer' }, year: 2019, reference: 'https://github.com/karpathy/nanoGPT',
    description: 'The same GPT plus a nanoGPT-style train.py: token batches, cross-entropy, AdamW, gradient clipping and a cosine schedule.',
    zhDescription: '同一个 GPT，加上 nanoGPT 风格的 train.py：token 批数据、交叉熵、AdamW、梯度裁剪和余弦学习率。',
    files: { 'model.py': gpt, 'train.py': gptTrain },
    experiments: [
      { ...exp('Batch 64', 'batch 改成 64', 'The batch dimension of every tensor in the step doubles.', '训练步里每个张量的 batch 维翻倍。', 'BATCH_SIZE = 32', 'BATCH_SIZE = 64'), file: 'train.py' },
      { ...exp('8 layers', '8 层', 'The model block grows; the training step stays the same.', '模型模块变大，训练步本身不变。', 'GPTConfig(n_layer=4, n_head=4, n_embd=128)\nmodel', 'GPTConfig(n_layer=8, n_head=4, n_embd=128)\nmodel'), file: 'train.py' },
    ],
  },
  {
    id: 'code-vit', name: 'Vision Transformer', zhName: '视觉 Transformer ViT', family: { en: 'Transformer', zh: 'Transformer' }, year: 2020, reference: 'https://arxiv.org/abs/2010.11929',
    description: 'Images as sequences of patch tokens with a [CLS] token and learned positions.',
    zhDescription: '把图像看作 patch token 序列，并加入 [CLS] token 和可学习位置编码。',
    files: { 'model.py': vit },
    experiments: [
      exp('Bigger patches', '更大的 patch', 'PATCH = 8 gives 16 tokens instead of 64.', 'PATCH = 8 时 token 从 64 个变成 16 个。', 'PATCH = 4', 'PATCH = 8'),
      exp('12 blocks', '12 层', 'A deeper encoder stack.', '更深的编码器堆叠。', 'DEPTH = 6', 'DEPTH = 12'),
      exp('5 heads (error)', '5 个头（会报错）', '192 is not divisible by 5 heads.', '192 不能被 5 个头整除。', 'HEADS = 3', 'HEADS = 5'),
    ],
  },
  {
    id: 'code-unet', name: 'U-Net', zhName: 'U-Net 分割网络', family: { en: 'CNN', zh: '卷积网络' }, year: 2015, reference: 'https://arxiv.org/abs/1505.04597',
    description: 'Encoder–decoder with skip connections; the widths list sets the number of levels.',
    zhDescription: '带跳跃连接的编码器–解码器；WIDTHS 列表决定层级数量。',
    files: { 'model.py': unet },
    experiments: [
      exp('Add a level', '增加一层', 'A fifth width adds one Down and one Up block.', '第五个宽度会多出一个 Down 与一个 Up 块。', 'WIDTHS = [32, 64, 128, 256]', 'WIDTHS = [32, 64, 128, 256, 512]'),
      exp('Odd size (error)', '奇数尺寸（会报错）', '100 → 50 → 25 → 12: upsampling gives 24, which cannot be concatenated with 25.', '100 → 50 → 25 → 12：上采样得到 24，无法与 25 拼接。', 'x = torch.randn(1, 3, 128, 128)', 'x = torch.randn(1, 3, 100, 100)'),
    ],
  },
  {
    id: 'code-dit', name: 'DiT', zhName: 'DiT 扩散 Transformer', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2022, reference: 'https://arxiv.org/abs/2212.09748', study: 'dit',
    description: 'Diffusion Transformer with adaLN-Zero conditioning on timestep and class.',
    zhDescription: '以时间步和类别为条件、采用 adaLN-Zero 调制的扩散 Transformer。',
    files: { 'model.py': dit },
    experiments: [
      exp('Patch size 4', 'Patch 改为 4', 'Fewer, larger patches: 64 tokens instead of 256.', '更少更大的 patch：token 从 256 个变成 64 个。', 'PATCH = 2', 'PATCH = 4'),
      exp('8 blocks', '8 层', 'Doubles the number of DiT blocks.', 'DiT 块数量翻倍。', 'DEPTH = 4', 'DEPTH = 8'),
    ],
  },
  {
    id: 'code-vae', name: 'Conv VAE', zhName: '卷积 VAE', family: { en: 'Generative', zh: '生成模型' }, year: 2013, reference: 'https://arxiv.org/abs/1312.6114',
    description: 'Variational autoencoder with μ / log σ² heads and the reparameterization trick.',
    zhDescription: '带 μ / log σ² 输出与重参数化技巧的变分自编码器。',
    files: { 'model.py': vae },
    experiments: [
      exp('Deeper encoder', '更深的编码器', 'A fourth channel width adds a down/up stage and shrinks the latent grid to 4×4.', '第四个通道宽度会多出一级下/上采样，潜空间网格缩小到 4×4。', 'CHANNELS = [32, 64, 128]', 'CHANNELS = [32, 64, 128, 256]'),
      exp('Bigger latent', '更大的潜变量', 'LATENT_DIM = 128 widens μ, log σ² and z.', 'LATENT_DIM = 128 会加宽 μ、log σ² 和 z。', 'LATENT_DIM = 32', 'LATENT_DIM = 128'),
    ],
  },
  {
    id: 'code-lstm', name: 'BiLSTM classifier', zhName: '双向 LSTM 文本分类', family: { en: 'RNN', zh: '循环网络' }, year: 1997,
    description: 'Embedding → 2-layer bidirectional LSTM → mean pooling → classifier.',
    zhDescription: '词嵌入 → 两层双向 LSTM → 时间平均 → 分类器。',
    files: { 'model.py': lstm },
    experiments: [
      exp('One direction (error)', '改为单向（会报错）', 'The LSTM outputs H instead of 2H features, so fc no longer matches.', 'LSTM 输出 H 而不是 2H 个特征，fc 不再匹配。', 'bidirectional=True', 'bidirectional=False'),
      exp('Longer sentences', '更长的句子', 'SEQ_LEN = 120 lengthens the sequence axis only.', 'SEQ_LEN = 120 只会拉长序列维度。', 'SEQ_LEN = 40', 'SEQ_LEN = 120'),
    ],
  },
  {
    id: 'code-ddpm', name: 'DDPM', zhName: 'DDPM 去噪扩散', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2020, reference: 'https://arxiv.org/abs/2006.11239', study: 'ddpm',
    description: 'One DDPM training step: noise an image to step t, predict the noise with a time-conditioned U-Net.',
    zhDescription: 'DDPM 训练一步：把图像加噪到第 t 步，再用带时间步条件的 U-Net 预测噪声。',
    files: { 'model.py': ddpm },
    experiments: [
      exp('Remove a skip (error)', '去掉跳跃连接（会报错）', 'The decoder block expects the encoder channels too, so its first GroupNorm no longer matches.', '解码块还需要编码器那一路的通道，去掉后第一层 GroupNorm 的通道数对不上。', 'h = block(torch.cat([h, skips.pop()], dim=1), emb)', 'h = block(h, emb)'),
      exp('Add a level', '增加一层', 'A fourth width adds one more step down and up the U.', '第四个宽度会让 U 形再深一层。', 'WIDTHS = [64, 128, 256]', 'WIDTHS = [64, 128, 256, 256]'),
      exp('64×64 images', '64×64 图像', 'Every feature map doubles; parameters stay the same.', '每张特征图都翻倍，参数量不变。', 'IMAGE_SIZE = 32', 'IMAGE_SIZE = 64'),
    ],
  },
  {
    id: 'code-latent-diffusion', name: 'Latent Diffusion', zhName: '潜空间扩散 Stable Diffusion', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2022, reference: 'https://arxiv.org/abs/2112.10752', study: 'latent-diffusion', depth: 1,
    description: 'Stable Diffusion in miniature: VAE encoder, text encoder, cross-attention U-Net on latents, VAE decoder.',
    zhDescription: '迷你 Stable Diffusion：VAE 编码器、文本编码器、在潜空间带交叉注意力的 U-Net、VAE 解码器。',
    files: { 'model.py': latentDiffusion },
    experiments: [
      exp('CLIP-length prompt', 'CLIP 长度的提示词', '77 text tokens: the keys and values of every cross-attention grow.', '77 个文本 token：每处交叉注意力的 K、V 都会变长。', 'TOKENS = 16', 'TOKENS = 77'),
      exp('256×256 images', '256×256 图像', 'The latent grows to 32×32; the U-Net weights stay the same.', '潜变量变成 32×32，U-Net 权重不变。', 'IMAGE_SIZE = 128', 'IMAGE_SIZE = 256'),
    ],
  },
  {
    id: 'code-ddim', name: 'DDIM sampler', zhName: 'DDIM 采样器', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2020, reference: 'https://arxiv.org/abs/2010.02502', study: 'ddim', depth: 0,
    description: 'Deterministic DDIM sampling unrolled: the same noise predictor is called once per jump.',
    zhDescription: '展开的确定性 DDIM 采样：每跳一步就调用一次同一个噪声预测器。',
    files: { 'model.py': ddim },
    experiments: [
      exp('5 jumps', '跳 5 步', 'Each extra timestep adds another call of the noise predictor.', '每多一个时间步，就多一次噪声预测器调用。', 'SAMPLE_STEPS = [999, 666, 333, 0]', 'SAMPLE_STEPS = [999, 800, 600, 400, 200, 0]'),
    ],
  },
  {
    id: 'code-cfg', name: 'Classifier-free guidance', zhName: '无分类器引导 CFG', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2022, reference: 'https://arxiv.org/abs/2207.12598', study: 'cfg', depth: 1,
    description: 'One denoiser run twice, with and without the label, then pushed away from the unconditional guess.',
    zhDescription: '同一个去噪器带标签、不带标签各跑一次，再把结果推离无条件预测。',
    files: { 'model.py': cfg },
    experiments: [
      exp('Deeper denoiser', '更深的去噪器', 'Four residual blocks in each of the two passes.', '两次调用中各有四个残差块。', 'ResBlock(WIDTH) for _ in range(2)', 'ResBlock(WIDTH) for _ in range(4)'),
    ],
  },
  {
    id: 'code-flow-matching', name: 'Flow Matching', zhName: 'Flow Matching 流匹配', family: { en: 'Diffusion', zh: '扩散模型' }, year: 2022, reference: 'https://arxiv.org/abs/2210.02747', study: 'flow-matching',
    description: 'Rectified flow on 2D points: interpolate noise and data, regress the velocity, take an Euler step.',
    zhDescription: '二维点上的整流流：插值噪声与数据，回归速度场，再走一步欧拉积分。',
    files: { 'model.py': flowMatching },
    experiments: [
      exp('3D points', '三维点', 'DIM = 3 changes only the first and last layers.', 'DIM = 3 只改变第一层和最后一层。', 'DIM = 2', 'DIM = 3'),
      exp('Wider field', '更宽的速度网络', 'HIDDEN = 512 quadruples the hidden weights.', 'HIDDEN = 512 让隐藏层权重变成四倍。', 'HIDDEN = 256', 'HIDDEN = 512'),
    ],
  },
  {
    id: 'code-pi0', name: 'π0', zhName: 'π0 视觉-语言-动作模型', family: { en: 'Robotics', zh: '具身智能' }, year: 2024, reference: 'https://arxiv.org/abs/2410.24164', study: 'pi0',
    description: 'Vision-language-action policy: a VLM prefix and an action expert that share attention, trained with flow matching.',
    zhDescription: '视觉-语言-动作策略：VLM 前缀与动作专家共享注意力，用流匹配训练。',
    files: { 'model.py': pi0 },
    experiments: [
      exp('Longer action chunk', '更长的动作块', 'HORIZON = 100 predicts two seconds more of motion in one pass.', 'HORIZON = 100 一次预测更长的一段动作。', 'HORIZON = 50 ', 'HORIZON = 100 '),
      exp('6 layers', '6 层', 'Every layer pairs a VLM block with an expert block.', '每一层都同时包含 VLM 块和动作专家块。', 'DEPTH = 3 ', 'DEPTH = 6 '),
    ],
  },
  {
    id: 'code-pi05', name: 'π0.5', zhName: 'π0.5 开放世界 VLA', family: { en: 'Robotics', zh: '具身智能' }, year: 2025, reference: 'https://arxiv.org/abs/2504.16054', study: 'pi05',
    description: 'π0 plus discrete state tokens, a subtask text head and AdaRMSNorm time conditioning in the action expert.',
    zhDescription: '在 π0 基础上加入离散状态 token、子任务文本输出头，以及动作专家里的 AdaRMSNorm 时间条件。',
    files: { 'model.py': pi05 },
    experiments: [
      exp('Longer subtask', '更长的子任务文本', 'SUBTASK_TOKENS = 12 lengthens both the prefix and the text loss.', 'SUBTASK_TOKENS = 12 会同时拉长前缀和文本损失。', 'SUBTASK_TOKENS = 6 ', 'SUBTASK_TOKENS = 12 '),
      exp('Finer state bins', '更细的状态分桶', '1024 bins enlarge the shared token table.', '1024 个分桶会扩大共享的 token 表。', 'STATE_BINS = 256 ', 'STATE_BINS = 1024 '),
    ],
  },
  {
    id: 'code-video-dit', name: 'Video DiT', zhName: '视频生成 DiT', family: { en: 'Video', zh: '视频生成' }, year: 2024, reference: 'https://arxiv.org/abs/2412.20404', study: 'video-generation',
    description: 'Text-to-video: 3D VAE, spatiotemporal Transformer with text cross-attention, flow-matching loss.',
    zhDescription: '文生视频：3D VAE、带文本交叉注意力的时空 Transformer、流匹配损失。',
    files: { 'model.py': videoDit },
    experiments: [
      exp('32 frames', '32 帧', 'Twice as many latent frames: temporal attention sequences double.', '潜空间帧数翻倍，时间注意力的序列长度也翻倍。', 'FRAMES = 16', 'FRAMES = 32'),
      exp('Bigger patches', '更大的 patch', 'PATCH = 4 gives 4× fewer tokens per frame.', 'PATCH = 4 时每帧 token 数减少到四分之一。', 'PATCH = 2', 'PATCH = 4'),
    ],
  },
  {
    id: 'code-causal-video', name: 'Causal video', zhName: '因果视频生成', family: { en: 'Video', zh: '视频生成' }, year: 2024, reference: 'https://arxiv.org/abs/2412.07772', study: 'causal-video',
    description: 'Block-causal video continuation: clean past frames and a noisy new frame under a frame-level causal mask.',
    zhDescription: '分块因果的视频续写：干净的历史帧与一帧带噪新帧，在帧级因果掩码下共同建模。',
    files: { 'model.py': causalVideo },
    experiments: [
      exp('More context', '更多历史帧', 'CONTEXT_FRAMES = 6 makes the causal mask and attention maps larger.', 'CONTEXT_FRAMES = 6 让因果掩码和注意力图都变大。', 'CONTEXT_FRAMES = 3', 'CONTEXT_FRAMES = 6'),
    ],
  },
  {
    id: 'code-uniadet', name: 'UniADet', zhName: 'UniADet 无语言通用异常检测', family: { en: 'Anomaly detection', zh: '异常检测' }, year: 2026, reference: 'https://arxiv.org/abs/2601.05552',
    description: 'Language-free zero- and few-shot anomaly detection: a frozen ViT, per-layer decoupled cls/seg weights instead of text, plus a normal-patch memory. Rebuilt from the paper; the official code (github.com/gaobb/UniADet) is not released yet.',
    zhDescription: '不依赖语言的零样本/少样本异常检测：冻结 ViT，用逐层解耦的分类/分割权重代替文本，再加正常 patch 记忆库。按论文复现，官方代码（github.com/gaobb/UniADet）尚未发布。',
    files: { 'model.py': uniadet },
    experiments: [
      exp('4-shot memory', '4 张正常提示图', 'K = 4 prompt images make the memory bank four times larger; learnable weights stay the same.', 'K = 4 时记忆库扩大到四倍，可学习权重不变。', 'SHOTS = 1 ', 'SHOTS = 4 '),
      exp('Two layers only', '只取两层', 'Fewer readout layers: fewer W_cls / W_seg heads and memory lookups.', '读出层变少：W_cls / W_seg 头和记忆库查找都随之减少。', 'FEATURES_LIST = [4, 5, 6, 7, 8]', 'FEATURES_LIST = [6, 8]'),
    ],
  },
  {
    id: 'code-visualad', name: 'VisualAD', zhName: 'VisualAD 无语言零样本异常检测', family: { en: 'Anomaly detection', zh: '异常检测' }, year: 2026, reference: 'https://github.com/7HHHHH/VisualAD',
    description: 'CVPR 2026 (arXiv 2603.07952): learnable anomaly and normal tokens inside a frozen CLIP ViT, refined per layer by spatial cross-attention (SCA) and scored against MLP-aligned patches (SAF). One training step of the official code.',
    zhDescription: 'CVPR 2026（arXiv 2603.07952）：在冻结的 CLIP ViT 中插入可学习的异常/正常 token，经逐层空间交叉注意力（SCA）增强，再与 MLP 校准后的 patch（SAF）比较打分。画的是官方代码的一个训练步。',
    files: { 'model.py': visualad },
    experiments: [
      exp('More anchors', '更多锚点', 'NUM_ANCHORS = 8 gives each SCA module more queries and a wider gate.', 'NUM_ANCHORS = 8 时每个 SCA 模块有更多 query，门控也更宽。', 'NUM_ANCHORS = 4', 'NUM_ANCHORS = 8'),
      exp('224×224 input', '224×224 输入', '224×224 gives a 16×16 patch grid: 256 patches per layer and four times larger anomaly maps.', '224×224 会得到 16×16 的 patch 网格：每层 256 个 patch，异常图大四倍。', 'IMAGE_SIZE = 112', 'IMAGE_SIZE = 224'),
    ],
  },
]

export const codeModelById = Object.fromEntries(codeModels.map((entry) => [entry.id, entry])) as Record<string, CodeModelEntry>

/** Study map id → runnable code model that implements it. */
export const codeForStudy: Record<string, string> = {
  ...Object.fromEntries(codeModels.filter((entry) => entry.study).map((entry) => [entry.study!, entry.id])),
  unet: 'code-ddpm',
}

export const blankTemplate = `import torch
import torch.nn as nn


class MyModel(nn.Module):
    """Write your own architecture here.
    zh: 在这里编写你自己的模型结构。"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.act(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


model = MyModel()
x = torch.randn(1, 3, 32, 32)
y = model(x)
`

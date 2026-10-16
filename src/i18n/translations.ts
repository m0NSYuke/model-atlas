import type { Locale, ModelIR, ModelNode, ModelNodeType } from '../core/model-ir/types'

const messages = {
  en: {
    subtitle: 'Architecture explorer', localWorkspace: 'Local workspace', importedWorkspace: 'Imported model',
    import: 'Import model', export: 'Export JSON', resetSelection: 'Clear selection', language: 'Language',
    modelTree: 'MODEL TREE', nodes: 'NODES', architecture: 'ARCHITECTURE', irConnected: 'IR CONNECTED',
    videoAutoencoder: 'Video autoencoder', modelView: 'MODEL VIEW', perspective: '3D VIEW', diagramView: 'DIAGRAM', fit: 'Fit', resetCamera: 'Reset camera',
    toggleView: 'Switch between 3D and a flat architecture diagram', view3d: '3D', viewDiagram: 'Diagram', trace: 'Trace', traceHint: 'Highlight direct inputs and outputs of the selected node',
    showDetails: 'Details', hideDetails: 'Compact', labelHint: 'Show or hide node types and tensor shapes', expandView: 'Expand view', closeExpanded: 'Close expanded view',
    dragOrbit: 'Drag to orbit', scrollZoom: 'Scroll to zoom', pan: 'Shift + drag to pan', tensor: 'TENSOR', module: 'MODULE', latentBranch: 'LATENT / BRANCH', edges: 'EDGES',
    inspector: 'MODULE INSPECTOR', tensorShapes: 'TENSOR SHAPES', input: 'INPUT', output: 'OUTPUT', parameters: 'PARAMETERS', dimensions: 'DIMENSIONS', featureBlock: 'FEATURE BLOCK', connections: 'CONNECTIONS', incoming: 'INPUTS FROM', outgoing: 'OUTPUTS TO', noConnections: 'No direct connections',
    source: 'SOURCE', lines: 'Lines', openSource: 'Open source', learningNotes: 'LEARNING NOTES', reference: 'REFERENCE', openPaper: 'Open primary source', inspectTitle: 'Inspect a module',
    inspectHint: 'Select a node in the model tree or 3D view to see its structure and learning notes.', noSelection: 'NO SELECTION',
    codeWorkspace: 'CODE WORKSPACE', sourceReady: 'SOURCE LINKED', sourceSnapshot: 'SOURCE SNAPSHOT',
    codeHint: 'Edit stride in blocks.py to update tensor shapes', importedCodeHint: 'Imported files can be edited, but the graph comes from the imported Model IR', importedIrHint: 'Imported Model IR · the JSON is read-only', irTab: 'model-ir.json',
    importTitle: 'Import model architecture', importHint: 'Choose a local Model IR JSON file. Nodes and edges will be checked before loading.',
    chooseFile: 'Choose JSON file', dropFile: 'or drop it here', importSupport: 'Accepts a Model IR object or an Atlas project with model and optional codeFiles.',
    importCancel: 'Cancel', importSuccess: 'Model loaded', importError: 'Import failed', exportSuccess: 'JSON exported',
    parseReady: 'SHAPES SYNCED', parseError: 'PARSER ERROR', reading: 'Reading file…', close: 'Close', loadingEditor: 'Loading editor…',
    collapse: 'Collapse', expand: 'Expand', architectureNav: 'Model architecture', demo: 'Demo',
    piCategory: 'π SERIES', diffusionCategory: 'DIFFUSION & METHODS', videoCategory: 'VIDEO GENERATION',
    architectureType: 'Architecture', methodType: 'Method', overviewType: 'Overview',
    catalogWorkspace: 'Study atlas', catalogCodeHint: 'Built-in study map · inspect the Model IR JSON',
    timePreserved: 'Temporal stride is 1: the time axis is no longer downsampled. The fixed decoder now produces more frames.',
    gesture: 'Hand control', startGesture: 'Start hand control', stopGesture: 'Stop hand control', cameraPreview: 'Camera preview', gestureSettings: 'Hand control settings',
    gestureStatus_off: 'Camera off', gestureStatus_starting: 'Requesting camera', gestureStatus_loading: 'Loading hand model', gestureStatus_searching: 'Show one hand', gestureStatus_tracking: 'Hand detected', gestureStatus_error: 'Hand control error',
    gestureError_permission: 'Camera permission was denied. Allow access in your browser and try again.', gestureError_notFound: 'No camera was found.', gestureError_inUse: 'The camera is in use by another app.', gestureError_unsupported: 'This browser cannot run camera hand tracking.', gestureError_timeout: 'Camera did not respond. Check browser permission or try another browser.', gestureError_model: 'The local hand model could not load.', gestureError_unknown: 'Could not start the camera.',
    gestureCamera: 'Camera', gestureHand: 'Active hand', gestureAnyHand: 'Either hand', gestureLeftHand: 'Left', gestureRightHand: 'Right', gestureSensitivity: 'Rotation sensitivity', gestureMirror: 'Mirror control', gestureDetail: 'MODULE DETAIL', gestureSubmodules: 'CHILD MODULES', gestureHint: 'Point to highlight · pinch to select · pinch and move to rotate · hold to inspect',
  },
  zh: {
    subtitle: '模型架构探索器', localWorkspace: '本地工作区', importedWorkspace: '已导入模型',
    import: '导入模型', export: '导出 JSON', resetSelection: '清除选择', language: '语言',
    modelTree: '模型树', nodes: '节点', architecture: '模型结构', irConnected: 'IR 已连接',
    videoAutoencoder: '视频自动编码器', modelView: '模型视图', perspective: '3D 视图', diagramView: '结构图', fit: '适配', resetCamera: '重置相机',
    toggleView: '切换 3D 与正视结构图', view3d: '3D', viewDiagram: '结构图', trace: '追踪', traceHint: '突出选中节点的直接输入和输出',
    showDetails: '详情', hideDetails: '简洁', labelHint: '显示或隐藏节点类型和张量形状', expandView: '放大视图', closeExpanded: '关闭放大视图',
    dragOrbit: '拖动旋转', scrollZoom: '滚轮缩放', pan: 'Shift + 拖动平移', tensor: '张量', module: '模块', latentBranch: '潜变量 / 分支', edges: '连接',
    inspector: '模块检查器', tensorShapes: '张量形状', input: '输入', output: '输出', parameters: '参数', dimensions: '维度', featureBlock: '特征模块', connections: '连接关系', incoming: '输入来自', outgoing: '输出到', noConnections: '没有直接连接',
    source: '源码', lines: '行', openSource: '打开源码', learningNotes: '学习笔记', reference: '参考资料', openPaper: '打开原始资料', inspectTitle: '查看模块',
    inspectHint: '在模型树或 3D 视图中选择节点，查看结构和学习笔记。', noSelection: '未选择',
    codeWorkspace: '代码工作区', sourceReady: '源码已关联', sourceSnapshot: '源码快照',
    codeHint: '修改 blocks.py 中的 stride，观察张量形状变化', importedCodeHint: '导入的文件可以编辑，但结构图来自导入的 Model IR', importedIrHint: '已导入 Model IR；JSON 为只读', irTab: 'model-ir.json',
    importTitle: '导入模型架构', importHint: '选择本地 Model IR JSON。加载前会校验节点与连接。',
    chooseFile: '选择 JSON 文件', dropFile: '或拖拽文件至此', importSupport: '支持 Model IR 对象，或含 model 与可选 codeFiles 的 Atlas 项目。',
    importCancel: '取消', importSuccess: '模型已加载', importError: '导入失败', exportSuccess: '已导出 JSON',
    parseReady: '形状已同步', parseError: '解析错误', reading: '正在读取文件…', close: '关闭', loadingEditor: '正在加载编辑器…',
    collapse: '收起', expand: '展开', architectureNav: '模型结构', demo: '示例',
    piCategory: 'π 系列', diffusionCategory: '扩散模型与方法', videoCategory: '视频生成',
    architectureType: '模型架构', methodType: '方法模块', overviewType: '全局流程',
    catalogWorkspace: '学习图谱', catalogCodeHint: '内置学习图谱；可查看 Model IR JSON',
    timePreserved: '时间步幅为 1：时间轴不再下采样。固定倍率的解码器因此会产生更多帧。',
    gesture: '手势控制', startGesture: '开启手势控制', stopGesture: '关闭手势控制', cameraPreview: '摄像头预览', gestureSettings: '手势设置',
    gestureStatus_off: '摄像头已关闭', gestureStatus_starting: '正在请求摄像头', gestureStatus_loading: '正在加载手部模型', gestureStatus_searching: '请将一只手放入画面', gestureStatus_tracking: '已识别手部', gestureStatus_error: '手势控制出错',
    gestureError_permission: '摄像头权限被拒绝。请在浏览器中允许后重试。', gestureError_notFound: '没有找到摄像头。', gestureError_inUse: '摄像头正被其他程序占用。', gestureError_unsupported: '当前浏览器不支持摄像头手势识别。', gestureError_timeout: '摄像头未响应，请检查浏览器权限或换用其他浏览器。', gestureError_model: '无法加载本地手部模型。', gestureError_unknown: '无法启动摄像头。',
    gestureCamera: '摄像头', gestureHand: '操作手', gestureAnyHand: '任意手', gestureLeftHand: '左手', gestureRightHand: '右手', gestureSensitivity: '旋转灵敏度', gestureMirror: '镜像控制', gestureDetail: '模块细节', gestureSubmodules: '子模块', gestureHint: '食指指向高亮 · 短捏选中 · 捏住移动旋转 · 长捏查看细节',
  },
} as const

export type MessageKey = keyof typeof messages.en
export function t(locale: Locale, key: MessageKey): string { return messages[locale][key] }

const vaeZh: Record<string, { name: string; description: string; what?: string; why?: string; notes?: string }> = {
  input: { name: '视频输入', description: '一批 RGB 视频帧。', what: '五维视频张量。', notes: '维度依次为批次、通道、时间、高度和宽度：[B, C, T, H, W]。' },
  encoder: { name: '编码器输出', description: '经过 3D 卷积、因果下采样和中间模块后的完整编码器输出。', what: '此节点汇总前面几个编码器子模块。', why: '得到的特征体积降低了时空计算开销。' },
  conv3d: { name: '3D 卷积', description: '提取时间与空间上的局部特征。', what: '三维卷积同时作用于 T、H、W。', notes: 'stride=(1,2,2) 保留全部 16 帧，高度和宽度减半。' },
  downsample: { name: '因果 3D 下采样', description: '缩小时间与空间分辨率的因果 3D 下采样模块。', what: '带步幅的因果卷积缩小特征体积。', why: '较小的 T/H/W 降低潜变量的存储与处理成本。', notes: 'stride=2 同时压缩 T、H、W；改为 stride=(1,2,2) 后，时间轴保持不变。' },
  mid: { name: '中间模块', description: '在估计潜变量分布前处理压缩后的特征。', what: '在编码器瓶颈处细化特征。' },
  moments: { name: '分布参数', description: '将编码特征投影为均值与对数方差。', what: '编码器预测高斯分布的参数。', why: '均值与方差让模型学到平滑且可采样的潜空间。' },
  mu: { name: 'μ / 均值', description: '近似后验分布的均值。', what: '潜变量分布的中心。' },
  logvar: { name: 'log σ² / 对数方差', description: '近似后验分布的对数方差。', what: '控制均值附近的不确定性。', why: '预测对数方差有助于保持数值稳定。' },
  sampling: { name: '采样', description: '使用重参数化技巧采样潜变量张量。', what: 'z = μ + σ · ε，其中 ε 是标准高斯噪声。', why: '将随机性与模型参数分离，使梯度可以穿过采样过程。' },
  latent: { name: '潜变量', description: '视频的紧凑时空表示。', what: '压缩后的视频特征体积。', notes: '与输入相比，时间维减半，两个空间维各缩小四倍。' },
  decoder: { name: '解码器', description: '将潜变量张量展开为视频帧。', what: '可学习的重建路径。', why: '逆转压缩过程，从潜变量生成视频帧。' },
  output: { name: '重建输出', description: '重建后的 RGB 视频。', what: '视频张量形式的 VAE 输出。' },
}

export function localizedNode(node: ModelNode, locale: Locale, modelId: string) {
  const supplied = node.localized?.[locale]
  const demo = locale === 'zh' && modelId === 'vae3d' ? vaeZh[node.id] : undefined
  return {
    name: supplied?.name ?? demo?.name ?? node.name,
    description: supplied?.description ?? demo?.description ?? node.description,
    learning: {
      what: supplied?.learning?.what ?? demo?.what ?? node.learning?.what,
      why: supplied?.learning?.why ?? demo?.why ?? node.learning?.why,
      notes: supplied?.learning?.notes ?? demo?.notes ?? node.learning?.notes,
      formula: supplied?.learning?.formula ?? node.learning?.formula,
    },
  }
}

export function localizedModel(model: ModelIR, locale: Locale) {
  return model.localized?.[locale]?.name ?? model.name
}

export function localizedModelDescription(model: ModelIR, locale: Locale) {
  return model.localized?.[locale]?.description ?? model.description
}

const chineseTypes: Record<ModelNodeType, string> = {
  tensor: '张量', conv2d: '2D 卷积', conv3d: '3D 卷积', linear: '线性层', norm: '归一化',
  attention: '注意力', mlp: '前馈网络', encoder: '编码器', decoder: '解码器',
  downsample: '下采样', upsample: '上采样', latent: '潜变量', sampling: '采样',
  split: '切分', concat: '拼接', generic: '通用模块', conv1d: '1D 卷积', embedding: '嵌入',
  activation: '激活', pool: '池化 / 归约', dropout: '随机失活', add: '相加', multiply: '相乘',
  matmul: '矩阵乘法', reshape: '变形', recurrent: '循环层', loss: '损失', parameter: '参数张量', module: '模块', optimizer: '优化器',
}

const englishTypes: Partial<Record<ModelNodeType, string>> = {
  conv1d: 'CONV 1D', conv2d: 'CONV 2D', conv3d: 'CONV 3D', mlp: 'MLP', matmul: 'MATMUL', pool: 'POOL / REDUCE',
}

export function typeName(locale: Locale, type: ModelNodeType): string {
  return locale === 'zh' ? chineseTypes[type] ?? type : englishTypes[type] ?? type.toUpperCase()
}

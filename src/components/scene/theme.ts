import type { ModelEdge, ModelNodeType } from '../../core/model-ir/types'
import { useUiStore, type ResolvedTheme } from '../../store/uiStore'

type TypePalette = Record<ModelNodeType, string>
type EdgePalette = Record<NonNullable<ModelEdge['type']>, string>

/** One color per kind of operation, shared by the 3D view, diagram, tree and legend (dark theme: luminous pastels). */
export const typeColor: TypePalette = {
  tensor: '#7dd3fc', parameter: '#f9a8d4', latent: '#c4b5fd', sampling: '#c4b5fd',
  conv1d: '#38bdf8', conv2d: '#38bdf8', conv3d: '#22d3ee', downsample: '#22d3ee', upsample: '#2dd4bf',
  linear: '#fbbf24', mlp: '#fb923c', embedding: '#f472b6', attention: '#a78bfa', recurrent: '#f87171',
  norm: '#5eead4', activation: '#a3e635', dropout: '#64748b', pool: '#60a5fa',
  add: '#e2e8f0', multiply: '#e2e8f0', matmul: '#c084fc', concat: '#e2e8f0', split: '#cbd5e1', reshape: '#94a3b8',
  loss: '#fb7185', optimizer: '#fdba74', encoder: '#60a5fa', decoder: '#34d399', module: '#818cf8', generic: '#94a3b8',
}

/** Light theme: the same hues, deepened so they hold up on white paper. */
const typeColorLight: TypePalette = {
  tensor: '#0ea5e9', parameter: '#db2777', latent: '#8b5cf6', sampling: '#8b5cf6',
  conv1d: '#0284c7', conv2d: '#0284c7', conv3d: '#0891b2', downsample: '#0891b2', upsample: '#0d9488',
  linear: '#f59e0b', mlp: '#f97316', embedding: '#ec4899', attention: '#7c3aed', recurrent: '#ef4444',
  norm: '#14b8a6', activation: '#65a30d', dropout: '#94a3b8', pool: '#2563eb',
  add: '#64748b', multiply: '#64748b', matmul: '#9333ea', concat: '#64748b', split: '#64748b', reshape: '#94a3b8',
  loss: '#e11d48', optimizer: '#ea580c', encoder: '#2563eb', decoder: '#059669', module: '#4f46e5', generic: '#64748b',
}

export const edgeColor: EdgePalette = {
  data: '#5eb6d8', residual: '#f0abfc', branch: '#a78bfa', condition: '#fcd34d', training: '#fdba74', feedback: '#fdba74', reuse: '#c4b5fd',
}

const edgeColorLight: EdgePalette = {
  data: '#6b8bb5', residual: '#c026d3', branch: '#7c3aed', condition: '#d97706', training: '#ea580c', feedback: '#ea580c', reuse: '#8b5cf6',
}

export interface ScenePalette {
  fog: string
  grid: { cell: string; section: string }
  hemisphere: [string, string, number]
  ambient: number
  key: number
  rim: string
  /** Fill of flat "card" blocks in study maps. */
  card: string
  cardActive: string
  stripe: string
  op: string
  outline: string
  /** Emissive glow multiplier: bright on dark backgrounds, subtle on light ones. */
  glow: number
  frames: string[]
}

export interface Palette { theme: ResolvedTheme; types: TypePalette; edges: EdgePalette; scene: ScenePalette }

export const palettes: Record<ResolvedTheme, Palette> = {
  dark: {
    theme: 'dark', types: typeColor, edges: edgeColor,
    scene: {
      fog: '#0b0e17', grid: { cell: '#1a2234', section: '#27324b' }, hemisphere: ['#c7d7ff', '#1b1630', 0.85], ambient: 0.55, key: 1.7, rim: '#a78bfa',
      card: '#151b2b', cardActive: '#1e2840', stripe: '#0b1220', op: '#e5e7eb', outline: '#ffffff', glow: 1,
      frames: ['#818cf8', '#a78bfa', '#f0abfc', '#5eead4'],
    },
  },
  light: {
    theme: 'light', types: typeColorLight, edges: edgeColorLight,
    scene: {
      fog: '#eef0f6', grid: { cell: '#dfe3ec', section: '#c9cfdd' }, hemisphere: ['#ffffff', '#c9cede', 1.1], ambient: 0.85, key: 1.9, rim: '#c4b5fd',
      card: '#ffffff', cardActive: '#f3f4ff', stripe: '#ffffff', op: '#475569', outline: '#1e1b4b', glow: 0.35,
      frames: ['#6366f1', '#8b5cf6', '#d946ef', '#14b8a6'],
    },
  },
}

/** The palette for the current theme; components re-render when the theme flips. */
export function usePalette(): Palette {
  return palettes[useUiStore((state) => state.resolved)] ?? palettes.dark
}

export const opSymbol: Partial<Record<string, string>> = {
  backward: '∇', step: '↻', clip_grad_norm: '‖∇‖', clip_grad_value: '∇±',
  add: '+', sub: '−', mul: '×', div: '÷', matmul: '@', einsum: 'Σ', concat: '⊕', stack: '⊞', chunk: '⫶', split: '⫶', mask: '◩', pow: '^',
}

export const legendTypes: { type: ModelNodeType; en: string; zh: string }[] = [
  { type: 'conv2d', en: 'Convolution', zh: '卷积' },
  { type: 'linear', en: 'Linear', zh: '全连接' },
  { type: 'attention', en: 'Attention', zh: '注意力' },
  { type: 'embedding', en: 'Embedding', zh: '嵌入' },
  { type: 'norm', en: 'Normalization', zh: '归一化' },
  { type: 'activation', en: 'Activation', zh: '激活' },
  { type: 'pool', en: 'Pool / reduce', zh: '池化 / 归约' },
  { type: 'upsample', en: 'Upsample', zh: '上采样' },
  { type: 'recurrent', en: 'Recurrent', zh: '循环' },
  { type: 'add', en: 'Tensor op', zh: '张量运算' },
  { type: 'loss', en: 'Loss', zh: '损失' },
  { type: 'optimizer', en: 'Backward / optimizer', zh: '反向传播 / 优化器' },
  { type: 'tensor', en: 'Tensor / I/O', zh: '张量 / 输入输出' },
  { type: 'parameter', en: 'Parameter', zh: '参数张量' },
  { type: 'module', en: 'Module (double-click to open)', zh: '模块（双击展开）' },
]

export function formatCount(value?: number): string {
  if (value === undefined) return '—'
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`
  if (value >= 1e6) return `${(value / 1e6).toFixed(value >= 1e7 ? 1 : 2)}M`
  if (value >= 1e3) return `${(value / 1e3).toFixed(value >= 1e4 ? 0 : 1)}K`
  return String(value)
}

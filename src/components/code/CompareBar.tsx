import { useState } from 'react'
import { ChevronDown, Eye, EyeOff, GitCompareArrows } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { useModelDiff, useStudyStore } from '../../store/studyStore'
import { localizedNode } from '../../i18n/translations'
import { formatCount } from '../scene/theme'
import { tr } from '../scene/useViewGraph'
import type { TensorShape } from '../../core/model-ir/types'

const shape = (value?: TensorShape) => value ? value.join('×') : '—'
const signed = (value: number) => `${value > 0 ? '+' : value < 0 ? '−' : '±'}${formatCount(Math.abs(value))}`

/** What the current edits did to the architecture, against the original code. */
export function CompareBar() {
  const diff = useModelDiff()
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  const selectNode = useModelStore((state) => state.selectNode)
  const compare = useStudyStore((state) => state.compare)
  const [open, setOpen] = useState(false)
  if (!diff) return null
  const delta = diff.totalTo - diff.totalFrom
  const percent = diff.totalFrom ? (delta / diff.totalFrom) * 100 : 0
  const name = (node: Parameters<typeof localizedNode>[0]) => localizedNode(node, locale, modelId).name
  return <div className={`compare-bar ${open ? 'open' : ''}`}>
    <div className="compare-head">
      <button type="button" className="compare-summary" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <GitCompareArrows size={13} />
        <b>{tr(locale, 'vs original', '对比原始代码')}</b>
        {diff.empty ? <span className="compare-chip">{tr(locale, 'no structural change', '结构无变化')}</span> : <>
          {diff.added.length > 0 && <span className="compare-chip added">+{diff.added.length} {tr(locale, 'new', '新增')}</span>}
          {diff.removed.length > 0 && <span className="compare-chip removed">−{diff.removed.length} {tr(locale, 'removed', '删除')}</span>}
          {diff.reshaped.length > 0 && <span className="compare-chip changed">{diff.reshaped.length} {tr(locale, 'reshaped', '形状变化')}</span>}
          {delta !== 0 && <span className={`compare-chip ${delta > 0 ? 'added' : 'removed'}`}>{tr(locale, 'params', '参数')} {signed(delta)} ({percent >= 0 ? '+' : ''}{percent.toFixed(Math.abs(percent) < 1 ? 2 : 1)}%)</span>}
        </>}
        <ChevronDown size={13} className="compare-chevron" />
      </button>
      <button type="button" className={`compare-eye ${compare ? 'on' : ''}`} onClick={() => useStudyStore.getState().setCompare(!compare)} title={tr(locale, 'Mark changed blocks in 3D and the diagram', '在 3D 和结构图中标出改动的模块')}>{compare ? <Eye size={13} /> : <EyeOff size={13} />}</button>
    </div>
    {open && !diff.empty && <div className="compare-list">
      {diff.added.slice(0, 30).map((node) => <button type="button" key={`a${node.id}`} onClick={() => selectNode(node.id)}><i className="added">+</i><span>{name(node)}</span><code>{shape(node.outputShape)}</code></button>)}
      {diff.reshaped.slice(0, 30).map(({ node, from, to }) => <button type="button" key={`r${node.id}`} onClick={() => selectNode(node.id)}><i className="changed">Δ</i><span>{name(node)}</span><code>{shape(from)} → {shape(to)}</code></button>)}
      {diff.params.filter(({ node }) => !diff.added.includes(node)).slice(0, 30).map(({ node, from, to }) => <button type="button" key={`p${node.id}`} onClick={() => selectNode(node.id)}><i className="changed">#</i><span>{name(node)}</span><code>{formatCount(from)} → {formatCount(to)}</code></button>)}
      {diff.removed.slice(0, 20).map((node) => <div key={`d${node.id}`} className="compare-removed"><i className="removed">−</i><span>{name(node)}</span><code>{shape(node.outputShape)}</code></div>)}
    </div>}
  </div>
}

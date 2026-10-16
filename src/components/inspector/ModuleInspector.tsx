import { useMemo } from 'react'
import { ArrowDown, BookOpen, Braces, NotebookPen, ChevronRight, Code2, CornerDownRight, ExternalLink, GitBranch, Map as MapIcon, MousePointerClick, PieChart, Sigma } from 'lucide-react'
import { getChildren, getNode } from '../../core/model-ir/utils'
import { useModelStore } from '../../store/modelStore'
import { localizedNode, t, typeName } from '../../i18n/translations'
import { studyById } from '../../data/models/studyCatalog'
import { codeForStudy, codeModelById } from '../../data/models/codeCatalog'
import type { ModelEdge, TensorShape } from '../../core/model-ir/types'
import { formatCount, legendTypes, usePalette } from '../scene/theme'
import { tr } from '../scene/useViewGraph'
import { ReadingGuide } from './ReadingGuide'
import { notesKey, useStudyStore } from '../../store/studyStore'

/** The reader's own note on a block: kept in this browser, carried by share links and exports. */
function NodeNote({ nodeId }: { nodeId: string }) {
  const locale = useModelStore((state) => state.locale)
  const key = useModelStore(notesKey)
  const note = useStudyStore((state) => state.notes[key]?.[nodeId] ?? '')
  return <section className="inspector-section note-section">
    <h2><NotebookPen size={13} />{tr(locale, 'My note', '我的笔记')}</h2>
    <textarea rows={note ? 3 : 2} value={note} placeholder={tr(locale, 'What this block does, a question, an idea to try… (shared with the link)', '这个模块做什么、疑问、想试的改法……（会随分享链接带上）')}
      onChange={(event) => useStudyStore.getState().setNote(key, nodeId, event.target.value)} />
  </section>
}

const edgeNames: Record<NonNullable<ModelEdge['type']>, { en: string; zh: string }> = {
  data: { en: 'data', zh: '数据' }, residual: { en: 'residual', zh: '残差' }, branch: { en: 'skip', zh: '跳连' },
  condition: { en: 'condition', zh: '条件' }, training: { en: 'training', zh: '训练' }, feedback: { en: 'feedback', zh: '回路' }, reuse: { en: 'shared', zh: '复用' },
}

/** Shape boxes with the axes that changed highlighted. */
function ShapeDiff({ input, output }: { input?: TensorShape; output?: TensorShape }) {
  const width = Math.max(input?.length ?? 0, output?.length ?? 0)
  const cells = (shape: TensorShape | undefined, other: TensorShape | undefined) => shape ? <div className="shape-cells">
    {shape.map((axis, index) => <span key={index} className={other && other.length === width && shape.length === width && other[index] !== axis ? 'changed' : ''}>{axis}</span>)}
  </div> : <div className="shape-cells empty">—</div>
  return <div className="shape-diff">
    {cells(input, output)}
    <ArrowDown size={14} />
    {cells(output, input)}
  </div>
}

/** Where the parameters live: one bar segment per layer type, sized by its share of the total. */
function Composition() {
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const { types } = usePalette()
  const parts = useMemo(() => {
    const groups = new Set(model.nodes.map((node) => node.parentId).filter(Boolean))
    const totals = new Map<string, number>()
    model.nodes.forEach((node) => { if (!groups.has(node.id) && node.paramCount) totals.set(node.type, (totals.get(node.type) ?? 0) + node.paramCount) })
    const sum = [...totals.values()].reduce((a, b) => a + b, 0)
    return { sum, list: [...totals.entries()].sort((a, b) => b[1] - a[1]).map(([type, value]) => ({ type: type as keyof typeof types, value, share: sum ? value / sum : 0 })) }
  }, [model])
  if (!parts.sum) return null
  const label = (type: string) => legendTypes.find((entry) => entry.type === type)?.[locale] ?? typeName(locale, type as keyof typeof types)
  return <section className="inspector-section composition">
    <h2><PieChart size={13} />{tr(locale, 'Where the parameters are', '参数分布')}</h2>
    <div className="composition-bar">{parts.list.map((part) => <i key={part.type} style={{ flexGrow: Math.max(part.share, 0.004), background: types[part.type] }} title={`${label(part.type)} · ${formatCount(part.value)}`} />)}</div>
    <div className="composition-list">{parts.list.slice(0, 6).map((part) => <div key={part.type}>
      <i style={{ background: types[part.type] }} /><span>{label(part.type)}</span><b>{(part.share * 100).toFixed(part.share >= 0.1 ? 0 : 1)}%</b><small>{formatCount(part.value)}</small>
    </div>)}</div>
  </section>
}

export function ModuleInspector() {
  const model = useModelStore((state) => state.modelIR)
  const selectedNodeId = useModelStore((state) => state.selectedNodeId)
  const locale = useModelStore((state) => state.locale)
  const codeFiles = useModelStore((state) => state.codeFiles)
  const modelOrigin = useModelStore((state) => state.modelOrigin)
  const builtInId = useModelStore((state) => state.builtInId)
  const openSource = useModelStore((state) => state.openSource)
  const selectNode = useModelStore((state) => state.selectNode)
  const node = getNode(model, selectedNodeId)
  const { types: typeColor, edges: edgeColor } = usePalette()
  const content = node ? localizedNode(node, locale, model.id) : null
  const reference = modelOrigin === 'import' ? undefined : builtInId ? codeModelById[builtInId]?.reference ?? studyById[builtInId]?.reference : undefined
  const name = (id: string) => { const item = getNode(model, id); return item ? localizedNode(item, locale, model.id).name : id }
  const children = node ? getChildren(model, node.id).filter((child) => child.type !== 'parameter') : []
  const incoming = node ? model.edges.filter((edge) => edge.target === node.id) : []
  const outgoing = node ? model.edges.filter((edge) => edge.source === node.id) : []
  const share = node?.paramCount && model.totalParams ? node.paramCount / model.totalParams : 0
  const ancestors: { id: string; name: string }[] = []
  for (let current = node?.parentId; current; current = getNode(model, current)?.parentId) ancestors.unshift({ id: current, name: name(current) })

  if (!node || !content) {
    const description = model.localized?.[locale]?.description ?? model.description
    return <div className="inspector-content inspector-empty">
      <div className="empty-hero">
        <span className="empty-icon"><MousePointerClick size={18} /></span>
        <div>
          <h2>{model.localized?.[locale]?.name ?? model.name}</h2>
          <p className="empty-sub">{tr(locale, 'Nothing selected — click any block to inspect it.', '尚未选中 · 点击任意模块查看详情')}</p>
        </div>
      </div>
      {description && <p className="inspector-description">{description}</p>}
      <ReadingGuide />
      <Composition />
      <ul className="tips">
        {modelOrigin === 'code' && <li><b>{tr(locale, 'Live code', '实时代码')}</b>{tr(locale, 'Edit the PyTorch on the left — the architecture rebuilds as you type.', '修改左侧 PyTorch 代码，结构会随输入实时重建。')}</li>}
        {modelOrigin === 'code' && <li><b>{tr(locale, 'Experiments', '实验')}</b>{tr(locale, 'Try the chips above the editor to see shapes and errors change.', '试试编辑器上方的实验按钮，观察形状和报错如何变化。')}</li>}
        <li><b>{tr(locale, 'Open modules', '展开模块')}</b>{tr(locale, 'Double-click a solid block, or use Depth to expand every level.', '双击实心模块，或用“层级”一次展开全部。')}</li>
        <li><b>{tr(locale, 'Jump anywhere', '快速跳转')}</b>{tr(locale, 'Press Ctrl+K and type a layer name.', '按 Ctrl+K，输入层名即可跳转。')}</li>
      </ul>
      <div className="overview-links">
        {modelOrigin === 'catalog' && builtInId && codeForStudy[builtInId] && <button type="button" className="reference-link" onClick={() => useModelStore.getState().loadBuiltIn(codeForStudy[builtInId])}><Code2 size={13} />{tr(locale, 'Open runnable code', '打开可运行代码')}</button>}
        {modelOrigin === 'code' && builtInId && codeModelById[builtInId]?.study && <button type="button" className="reference-link" onClick={() => useModelStore.getState().loadBuiltIn(codeModelById[builtInId].study!)}><MapIcon size={13} />{tr(locale, 'Open the study map', '查看学习图谱')}</button>}
        {reference && <a className="reference-link" href={reference} target="_blank" rel="noreferrer"><ExternalLink size={13} />{t(locale, 'openPaper')}</a>}
      </div>
    </div>
  }

  const color = typeColor[node.type] ?? '#94a3b8'
  const config = Object.entries(node.parameters ?? {})

  return <div className="inspector-panel" style={{ ['--accent' as string]: color }}>
    <div className="inspector-content" key={node.id}>
      {ancestors.length > 0 && <nav className="crumbs">{ancestors.map((item) => <button type="button" key={item.id} onClick={() => selectNode(item.id)}>{item.name}<ChevronRight size={11} /></button>)}</nav>}
      <div className="inspector-type"><i style={{ background: color }} />{typeName(locale, node.type)}{node.className && <code>{node.className}</code>}</div>
      <h1 className="inspector-title">{content.name}</h1>
      {content.description && <p className="inspector-description">{content.description}</p>}
      <NodeNote nodeId={node.id} />

      {(node.paramCount !== undefined || children.length > 0) && <div className="stat-tiles">
        {node.paramCount !== undefined && <div><b>{formatCount(node.paramCount)}</b><span>{tr(locale, 'parameters', '参数')}</span></div>}
        {share > 0 && <div><b>{share >= 0.001 ? `${(share * 100).toFixed(share > 0.1 ? 0 : 1)}%` : '<0.1%'}</b><span>{tr(locale, 'of the model', '占全模型')}</span><i style={{ width: `${Math.max(2, share * 100)}%` }} /></div>}
        {children.length > 0 && <div><b>{children.length}</b><span>{tr(locale, 'children', '子模块')}</span></div>}
      </div>}

      {(node.inputShape || node.outputShape) && <section className="inspector-section">
        <h2><Sigma size={13} />{t(locale, 'tensorShapes')}</h2>
        <ShapeDiff input={node.inputShape} output={node.outputShape} />
      </section>}

      {config.length > 0 && <section className="inspector-section">
        <h2><Braces size={13} />{tr(locale, 'Configuration', '配置')}</h2>
        <div className="parameter-list">{config.map(([key, value]) => <div className="parameter-row" key={key}><span>{key}</span><code>{Array.isArray(value) ? `(${value.join(', ')})` : String(value)}</code></div>)}</div>
      </section>}

      {(node.source || node.callSite) && <section className="inspector-section">
        <h2><Code2 size={13} />{t(locale, 'source')}</h2>
        <div className="source-links">
          {node.source && node.source.file in codeFiles && <button type="button" onClick={() => openSource(node.source!.file, node.source!.startLine)}>
            <span>{tr(locale, 'Defined at', '定义于')}</span><code>{node.source.file}:{node.source.startLine}</code>
          </button>}
          {node.callSite && node.callSite.file in codeFiles && <button type="button" onClick={() => openSource(node.callSite!.file, node.callSite!.line)}>
            <span>{tr(locale, 'Called in forward', 'forward 中调用')}</span><code>{node.callSite.file}:{node.callSite.line}</code>
          </button>}
          {node.source && !(node.source.file in codeFiles) && <div><span>{node.source.file}</span><code>{t(locale, 'lines')} {node.source.startLine}–{node.source.endLine}</code></div>}
        </div>
      </section>}

      {(content.learning.what || content.learning.why || content.learning.notes || content.learning.formula) && <section className="inspector-section learning-section">
        <h2><BookOpen size={13} />{t(locale, 'learningNotes')}</h2>
        {content.learning.what && content.learning.what !== content.description && <p>{content.learning.what}</p>}
        {content.learning.why && <p className="why">{content.learning.why}</p>}
        {content.learning.formula && <div className="formula">{content.learning.formula}</div>}
        {content.learning.notes && <p className="notes">{content.learning.notes}</p>}
        {node.id === 'downsample' && Array.isArray(node.parameters?.stride) && node.parameters.stride[0] === 1 && <p className="shape-insight">{t(locale, 'timePreserved')}</p>}
      </section>}

      {children.length > 0 && <section className="inspector-section">
        <h2><CornerDownRight size={13} />{tr(locale, 'Inside this module', '模块内部')}</h2>
        <div className="child-list">{children.slice(0, 40).map((child) => <button type="button" key={child.id} onClick={() => selectNode(child.id)}>
          <i style={{ background: typeColor[child.type] }} /><span>{name(child.id)}</span>{child.paramCount ? <small>{formatCount(child.paramCount)}</small> : <small>{child.outputShape?.join('×')}</small>}
        </button>)}</div>
      </section>}

      {incoming.length + outgoing.length > 0 && <section className="inspector-section">
        <h2><GitBranch size={13} />{t(locale, 'connections')}</h2>
        {[{ list: incoming, key: 'incoming' as const, pick: (edge: ModelEdge) => edge.source }, { list: outgoing, key: 'outgoing' as const, pick: (edge: ModelEdge) => edge.target }].map(({ list, key, pick }) => list.length > 0 && <div className="connection-group" key={key}>
          <div className="connection-heading">{t(locale, key)}<span>{list.length}</span></div>
          {list.slice(0, 24).map((edge) => <button type="button" key={edge.id} className="connection-row" onClick={() => selectNode(pick(edge))}>
            <span className="connection-name">{name(pick(edge))}</span>
            <span className="connection-type" style={{ color: edgeColor[edge.type ?? 'data'] }}>{edgeNames[edge.type ?? 'data'][locale]}</span>
          </button>)}
        </div>)}
      </section>}

      {reference && <a className="reference-link" href={reference} target="_blank" rel="noreferrer"><ExternalLink size={13} />{t(locale, 'openPaper')}</a>}
    </div>
  </div>
}

import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Search, X } from 'lucide-react'
import type { ModelNode } from '../../core/model-ir/types'
import { collapsedPredicate, useModelStore } from '../../store/modelStore'
import { localizedNode, t, typeName } from '../../i18n/translations'
import { formatCount, usePalette } from '../scene/theme'
import { tr } from '../scene/useViewGraph'

function Row({ node, depth, childCount, expanded, onToggle }: { node: ModelNode; depth: number; childCount: number; expanded: boolean; onToggle?: () => void }) {
  const locale = useModelStore((state) => state.locale)
  const modelId = useModelStore((state) => state.modelIR.id)
  const selected = useModelStore((state) => state.selectedNodeId === node.id)
  const hovered = useModelStore((state) => state.hoveredNodeId === node.id)
  const selectNode = useModelStore((state) => state.selectNode)
  const hoverNode = useModelStore((state) => state.hoverNode)
  const ref = useRef<HTMLDivElement>(null)
  const typeColor = usePalette().types
  useEffect(() => { if (selected) ref.current?.scrollIntoView({ block: 'nearest' }) }, [selected])
  const name = localizedNode(node, locale, modelId).name
  const quiet = node.type === 'parameter' || node.type === 'tensor' && node.className !== 'Input' && node.className !== 'Output'
  return <div ref={ref} className={`tree-row ${selected ? 'selected' : ''} ${hovered ? 'hovered' : ''} ${quiet ? 'quiet' : ''}`} style={{ paddingLeft: 8 + depth * 14 }}
    onMouseEnter={() => hoverNode(node.id)} onMouseLeave={() => hoverNode(null)}>
    <button type="button" className="tree-expander" onClick={onToggle} disabled={!childCount} aria-label={`${t(locale, expanded ? 'collapse' : 'expand')} ${name}`}>
      {childCount ? expanded ? <ChevronDown size={13} /> : <ChevronRight size={13} /> : null}
    </button>
    <button type="button" className="tree-node-button" onClick={() => selectNode(node.id)} title={`${typeName(locale, node.type)}${node.className ? ` · ${node.className}` : ''}`}>
      <i className="tree-dot" style={{ background: typeColor[node.type], boxShadow: childCount ? `0 0 0 2px var(--surface), 0 0 0 3.5px ${typeColor[node.type]}` : undefined }} />
      <span className="tree-node-name">{name}</span>
      {node.className && node.className !== name && <span className="tree-class">{node.className}</span>}
      {node.paramCount ? <span className="tree-params">{formatCount(node.paramCount)}</span> : childCount && !expanded ? <span className="tree-params">{childCount}</span> : null}
    </button>
  </div>
}

export function ModelTree() {
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const depth = useModelStore((state) => state.depth)
  const toggles = useModelStore((state) => state.toggles)
  const toggleCollapsed = useModelStore((state) => state.toggleCollapsed)
  const [query, setQuery] = useState('')
  const [localClosed, setLocalClosed] = useState<Record<string, boolean>>({})
  const [showParams, setShowParams] = useState(false)
  const isCollapsed = useMemo(() => collapsedPredicate({ modelIR: model, depth, toggles }), [model, depth, toggles])
  const children = useMemo(() => {
    const map = new Map<string | undefined, ModelNode[]>()
    model.nodes.forEach((node) => {
      if (!showParams && node.type === 'parameter') return
      map.set(node.parentId, [...(map.get(node.parentId) ?? []), node])
    })
    return map
  }, [model, showParams])
  const isCode = model.layout === 'code'
  const expandedOf = (id: string) => isCode ? !isCollapsed(id) : !localClosed[id]
  const toggle = (id: string) => isCode ? toggleCollapsed(id) : setLocalClosed((current) => ({ ...current, [id]: !current[id] }))

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return null
    return model.nodes.filter((node) => [node.name, node.className, node.id, localizedNode(node, locale, model.id).name].some((text) => text?.toLowerCase().includes(needle))).slice(0, 200)
  }, [query, model, locale])

  const rows: { node: ModelNode; depth: number }[] = []
  const walk = (parent: string | undefined, level: number) => {
    for (const node of children.get(parent) ?? []) {
      rows.push({ node, depth: level })
      if ((children.get(node.id) ?? []).length && expandedOf(node.id)) walk(node.id, level + 1)
    }
  }
  if (!matches) walk(undefined, 0)

  return <div className="tree-panel">
    <div className="tree-tools">
      <div className="tree-search">
        <Search size={13} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr(locale, 'Search layers…', '搜索层或模块…')} />
        {query && <button type="button" onClick={() => setQuery('')} aria-label={t(locale, 'close')}><X size={12} /></button>}
      </div>
      {isCode && <button type="button" className={`chip-toggle ${showParams ? 'on' : ''}`} aria-pressed={showParams} onClick={() => setShowParams((value) => !value)} data-tip={tr(locale, 'Show weight tensors such as positional embeddings', '显示位置编码等独立的权重张量')} data-tip-side="bottom">{tr(locale, 'Weights', '权重')}</button>}
    </div>
    <nav className="tree-list" aria-label={t(locale, 'architectureNav')}>
      {matches
        ? matches.length ? matches.map((node) => <Row key={node.id} node={node} depth={0} childCount={0} expanded={false} />) : <p className="tree-empty">{tr(locale, 'No matching layers', '没有匹配的层')}</p>
        : rows.map(({ node, depth: level }) => {
          const count = (children.get(node.id) ?? []).length
          return <Row key={node.id} node={node} depth={level} childCount={count} expanded={expandedOf(node.id)} onToggle={count ? () => toggle(node.id) : undefined} />
        })}
    </nav>
  </div>
}

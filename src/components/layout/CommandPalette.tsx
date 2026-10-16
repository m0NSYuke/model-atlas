import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { Box, Code2, CornerDownLeft, Download, FileCode2, FilePlus2, Focus, Languages, Layers3, LayoutGrid, ListTree, Map as MapIcon, Maximize2, Moon, Rotate3d, Route, Search, Sparkles, Sun, Tag, Upload, Waypoints } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { codeModels } from '../../data/models/codeCatalog'
import { studyCatalog } from '../../data/models/studyCatalog'
import { localizedModel, localizedNode, typeName } from '../../i18n/translations'
import { tr } from '../scene/useViewGraph'
import { formatCount, usePalette } from '../scene/theme'
import { exportProject } from './projectActions'
import { toggleSideDock, toggleSideSection } from './Rail'
import { useDialogFocus } from './useDialogFocus'

interface Item { id: string; group: string; label: string; detail?: string; keys?: string; icon: ReactNode; run: () => void; search: string }

/** Ctrl+K: run any action, open any model, or jump to any layer by name. */
export function CommandPalette({ onClose }: { onClose: () => void }) {
  const locale = useModelStore((state) => state.locale)
  const model = useModelStore((state) => state.modelIR)
  const { types } = usePalette()
  const [query, setQuery] = useState('')
  const [index, setIndex] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const dialog = useRef<HTMLDivElement>(null)
  // Before the input takes focus below, so the dialog remembers what to return focus to.
  useDialogFocus(dialog, onClose)

  const items = useMemo<Item[]>(() => {
    const ui = useUiStore.getState
    const store = useModelStore.getState
    const actions = tr(locale, 'Actions', '操作')
    const action = (id: string, en: string, zh: string, icon: ReactNode, run: () => void, keys?: string): Item => ({ id, group: actions, label: tr(locale, en, zh), keys, icon, run, search: `${en} ${zh}`.toLowerCase() })
    const resolved = ui().resolved
    const list: Item[] = [
      action('theme', resolved === 'dark' ? 'Switch to light mode' : 'Switch to dark mode', resolved === 'dark' ? '切换到明亮模式' : '切换到黑暗模式', resolved === 'dark' ? <Sun size={15} /> : <Moon size={15} />, () => ui().cycleTheme(), 'T'),
      action('code', 'Show / hide code panel', '显示 / 收起代码栏', <FileCode2 size={15} />, () => ui().togglePanel('codeOpen'), '['),
      action('side', 'Show / hide tree and inspector', '显示 / 收起模型树与检查器', <ListTree size={15} />, toggleSideDock, ']'),
      action('inspector', 'Show / hide module inspector', '显示 / 收起模块检查器', <Layers3 size={15} />, () => toggleSideSection('inspectorOpen')),
      action('focus', 'Focus mode', '专注模式（收起全部侧栏）', <Maximize2 size={15} />, () => ui().toggleFocusMode(), '\\'),
      action('3d', 'Show 3D view', '切换到 3D 视图', <Box size={15} />, () => ui().setView('3d'), '1'),
      action('2d', 'Show flat diagram', '切换到平面结构图', <Waypoints size={15} />, () => ui().setView('2d'), '2'),
      action('fit', 'Fit model in view', '适配视野', <Focus size={15} />, () => ui().fit(), 'F'),
      action('angle', 'Toggle angled / front camera', '切换斜视 / 正视', <Rotate3d size={15} />, () => ui().setAngle(ui().angle === 'iso' ? 'front' : 'iso'), 'V'),
      action('labels', 'Cycle labels', '切换标签显示', <Tag size={15} />, () => ui().cycleLabels(), 'L'),
      action('trace', 'Toggle connection tracing', '开关连接追踪', <Route size={15} />, () => ui().setTrace(!ui().trace), 'R'),
      action('flow', 'Toggle tensor flow', '开关张量流动', <Sparkles size={15} />, () => ui().setFlow(!ui().flow), 'P'),
      action('library', 'Open model library', '打开模型库', <LayoutGrid size={15} />, () => store().setLibraryOpen(true)),
      action('blank', 'New blank model', '新建空白模型', <FilePlus2 size={15} />, () => store().loadBlank()),
      action('import', 'Import .py / JSON', '导入 .py / JSON', <Upload size={15} />, () => ui().setOverlay('import')),
      action('export', 'Export project JSON', '导出项目 JSON', <Download size={15} />, exportProject),
      action('lang', locale === 'zh' ? 'Switch to English' : 'Switch to Chinese', locale === 'zh' ? '切换到英文' : '切换到中文', <Languages size={15} />, () => store().setLocale(locale === 'zh' ? 'en' : 'zh')),
    ]
    const models = tr(locale, 'Models', '模型')
    codeModels.forEach((entry) => list.push({ id: `model:${entry.id}`, group: models, label: locale === 'zh' ? entry.zhName : entry.name, detail: entry.family[locale], icon: <Code2 size={15} />, run: () => store().loadBuiltIn(entry.id), search: `${entry.name} ${entry.zhName} ${entry.family.en} ${entry.family.zh}`.toLowerCase() }))
    studyCatalog.forEach((entry) => list.push({ id: `study:${entry.model.id}`, group: models, label: localizedModel(entry.model, locale), detail: tr(locale, 'Study map', '学习图谱'), icon: <MapIcon size={15} />, run: () => store().loadBuiltIn(entry.model.id), search: `${entry.model.name} ${localizedModel(entry.model, 'zh')}`.toLowerCase() }))
    return list
  }, [locale])

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return items.filter((item) => item.group !== tr(locale, 'Models', '模型') || item.id.startsWith('model:'))
    const words = needle.split(/\s+/)
    const matched = items.filter((item) => words.every((word) => item.search.includes(word) || item.label.toLowerCase().includes(word)))
    const byId = new Map(model.nodes.map((node) => [node.id, node]))
    const parentName = (id?: string) => { const parent = id ? byId.get(id) : undefined; return parent ? localizedNode(parent, locale, model.id).name : undefined }
    const layers = model.nodes.filter((node) => node.type !== 'parameter').map((node) => ({ node, name: localizedNode(node, locale, model.id).name }))
      .filter(({ node, name }) => words.every((word) => [name, node.name, node.className, node.id].some((text) => text?.toLowerCase().includes(word))))
      .slice(0, 40)
      .map(({ node, name }): Item => ({
        id: `node:${node.id}`, group: tr(locale, 'Layers in this model', '当前模型中的层'), label: name,
        detail: [parentName(node.parentId), node.className ?? typeName(locale, node.type), node.outputShape?.join('×'), node.paramCount ? formatCount(node.paramCount) : undefined].filter(Boolean).join(' · '),
        icon: <i className="palette-dot" style={{ background: types[node.type] }} />,
        run: () => { useModelStore.getState().selectNode(node.id); const ui = useUiStore.getState(); if (ui.sideOpen && !ui.inspectorOpen) toggleSideSection('inspectorOpen') }, search: '',
      }))
    return [...matched, ...layers]
  }, [items, query, model, locale, types])

  useEffect(() => { setIndex(0) }, [query])
  useEffect(() => { input.current?.focus() }, [])
  useEffect(() => { list.current?.querySelector('[data-active="true"]')?.scrollIntoView({ block: 'nearest' }) }, [index])

  const run = (item: Item | undefined) => { if (!item) return; onClose(); item.run() }

  let lastGroup = ''
  return <div className="dialog-backdrop palette-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div ref={dialog} className="palette" role="dialog" aria-modal="true" tabIndex={-1} aria-label={tr(locale, 'Command palette', '命令面板')}>
      <div className="palette-input">
        <Search size={17} />
        <input ref={input} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tr(locale, 'Search actions, models or layers…', '搜索操作、模型或层…')}
          onKeyDown={(event) => {
            if (event.key === 'ArrowDown') { event.preventDefault(); setIndex((value) => Math.min(results.length - 1, value + 1)) }
            if (event.key === 'ArrowUp') { event.preventDefault(); setIndex((value) => Math.max(0, value - 1)) }
            if (event.key === 'Enter') { event.preventDefault(); run(results[index]) }
            if (event.key === 'Escape') { event.preventDefault(); onClose() }
          }} />
        <kbd>Esc</kbd>
      </div>
      <div className="palette-list" ref={list} role="listbox">
        {results.length === 0 && <div className="palette-empty">{tr(locale, 'Nothing matches', '没有匹配的结果')}</div>}
        {results.map((item, position) => {
          const heading = item.group !== lastGroup ? item.group : null
          lastGroup = item.group
          return <div key={item.id}>
            {heading && <div className="palette-group">{heading}</div>}
            <button type="button" role="option" aria-selected={position === index} data-active={position === index} className="palette-item"
              onMouseMove={() => setIndex(position)} onClick={() => run(item)}>
              <span className="palette-icon">{item.icon}</span>
              <span className="palette-label">{item.label}</span>
              {item.detail && <span className="palette-detail">{item.detail}</span>}
              {item.keys && <kbd>{item.keys}</kbd>}
              {position === index && <CornerDownLeft size={13} className="palette-enter" />}
            </button>
          </div>
        })}
      </div>
      <footer className="palette-foot">
        <span><kbd>↑</kbd><kbd>↓</kbd>{tr(locale, 'navigate', '选择')}</span>
        <span><kbd>Enter</kbd>{tr(locale, 'run', '执行')}</span>
        <span className="palette-foot-tip">{tr(locale, 'Tip: type a layer name like “attn” to jump to it', '提示：输入层名（如 “attn”）可直接跳转')}</span>
      </footer>
    </div>
  </div>
}

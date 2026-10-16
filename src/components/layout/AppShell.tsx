import { lazy, Suspense, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronDown, Layers3, ListTree, PanelLeftClose, PanelRightClose } from 'lucide-react'
import { ModelTree } from '../model-tree/ModelTree'
import { ModelScene } from '../scene/ModelScene'
import { ModuleInspector } from '../inspector/ModuleInspector'
import { useModelStore } from '../../store/modelStore'
import { defaultWidths, useUiStore } from '../../store/uiStore'
import { getNode } from '../../core/model-ir/utils'
import { typeName } from '../../i18n/translations'
import { tr } from '../scene/useViewGraph'
import { usePalette } from '../scene/theme'
import { Rail, toggleSideSection } from './Rail'
import { ModelLibrary } from './ModelLibrary'
import { ImportModelDialog } from './ImportModelDialog'
import { CommandPalette } from './CommandPalette'
import { ShortcutsOverlay } from './ShortcutsOverlay'
import { useShortcuts } from './useShortcuts'

// The code editor (Monaco) is the largest part of the app, so it loads separately and only once the code panel is shown.
const CodeWorkspace = lazy(() => import('../code/CodeWorkspace').then((module) => ({ default: module.CodeWorkspace })))

/** Drag handle on a dock edge. Reports the pointer's horizontal travel since the drag began. */
function ResizeHandle({ side, onDrag, onReset, label }: { side: 'left' | 'right'; onDrag: (delta: number, start: number) => void; onReset: () => void; label: string }) {
  const start = useRef<{ x: number; width: number } | null>(null)
  return <div className={`dock-resizer ${side}`} role="separator" aria-orientation="vertical" aria-label={label} title={label}
    onDoubleClick={onReset}
    onPointerDown={(event) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      const dock = event.currentTarget.parentElement!
      start.current = { x: event.clientX, width: dock.getBoundingClientRect().width }
      document.body.classList.add('is-resizing')
    }}
    onPointerMove={(event) => { if (start.current) onDrag(event.clientX - start.current.x, start.current.width) }}
    onPointerUp={() => { start.current = null; document.body.classList.remove('is-resizing') }}
    onPointerCancel={() => { start.current = null; document.body.classList.remove('is-resizing') }} />
}

function DockSection({ title, icon, open, onToggle, extra, children, style }: { title: string; icon: ReactNode; open: boolean; onToggle: () => void; extra?: ReactNode; children: ReactNode; style?: React.CSSProperties }) {
  return <section className={`dock-section ${open ? 'open' : 'closed'}`} style={style}>
    <header className="section-head">
      <button type="button" className="section-toggle" onClick={onToggle} aria-expanded={open}>
        <ChevronDown size={14} className="section-chevron" />
        {icon}<span>{title}</span>
      </button>
      {extra}
    </header>
    {open && <div className="section-body">{children}</div>}
  </section>
}

/** Vertical splitter between the tree and the inspector. */
function SectionSplitter() {
  const ref = useRef<HTMLDivElement>(null)
  const dragging = useRef(false)
  return <div ref={ref} className="section-splitter" role="separator" aria-orientation="horizontal"
    onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); dragging.current = true; document.body.classList.add('is-resizing-y') }}
    onPointerMove={(event) => {
      if (!dragging.current) return
      const container = ref.current?.parentElement?.getBoundingClientRect()
      if (container) useUiStore.getState().setSplit((event.clientY - container.top) / container.height)
    }}
    onPointerUp={() => { dragging.current = false; document.body.classList.remove('is-resizing-y') }}
    onDoubleClick={() => useUiStore.getState().setSplit(0.42)} />
}

function SideDock() {
  const locale = useModelStore((state) => state.locale)
  const model = useModelStore((state) => state.modelIR)
  const selected = useModelStore((state) => getNode(state.modelIR, state.selectedNodeId))
  const treeOpen = useUiStore((state) => state.treeOpen)
  const inspectorOpen = useUiStore((state) => state.inspectorOpen)
  const split = useUiStore((state) => state.split)
  const { types } = usePalette()
  const both = treeOpen && inspectorOpen
  return <div className="side-inner">
    <DockSection title={tr(locale, 'Model tree', '模型树')} icon={<ListTree size={14} />} open={treeOpen} onToggle={() => toggleSideSection('treeOpen')}
      style={both ? { flex: `0 0 ${split * 100}%` } : treeOpen ? { flex: '1 1 auto' } : undefined}
      extra={<>
        <span className="section-count">{model.nodes.length}</span>
        <button type="button" className="icon-button small" onClick={() => useUiStore.getState().togglePanel('sideOpen', false)} data-tip={tr(locale, 'Hide panel  ]', '收起侧栏  ]')} data-tip-side="left" aria-label={tr(locale, 'Hide panel', '收起侧栏')}><PanelRightClose size={15} /></button>
      </>}>
      <ModelTree />
    </DockSection>
    {both && <SectionSplitter />}
    <DockSection title={tr(locale, 'Module inspector', '模块检查器')} icon={<Layers3 size={14} />} open={inspectorOpen} onToggle={() => toggleSideSection('inspectorOpen')}
      style={inspectorOpen ? { flex: '1 1 0' } : undefined}
      extra={selected ? <span className="section-pill" style={{ ['--accent' as string]: types[selected.type] }}><i />{typeName(locale, selected.type)}</span> : null}>
      <ModuleInspector />
    </DockSection>
  </div>
}

export function AppShell() {
  const locale = useModelStore((state) => state.locale)
  const libraryOpen = useModelStore((state) => state.libraryOpen)
  const setLibraryOpen = useModelStore((state) => state.setLibraryOpen)
  const codeOpen = useUiStore((state) => state.codeOpen)
  const sideOpen = useUiStore((state) => state.sideOpen)
  const codeWidth = useUiStore((state) => state.codeWidth)
  const sideWidth = useUiStore((state) => state.sideWidth)
  const overlay = useUiStore((state) => state.overlay)
  const setWidth = useUiStore((state) => state.setWidth)
  const setOverlay = useUiStore((state) => state.setOverlay)
  const [mounted, setMounted] = useState(false)
  useShortcuts()
  // Skip the dock animation on first paint so restored layouts appear instantly.
  useEffect(() => { const frame = requestAnimationFrame(() => setMounted(true)); return () => cancelAnimationFrame(frame) }, [])
  const closeOverlay = useCallback(() => setOverlay(null), [setOverlay])
  // Once shown, the editor stays mounted so hiding the panel keeps its undo history and scroll position.
  const [codeShown, setCodeShown] = useState(codeOpen)
  useEffect(() => { if (codeOpen) setCodeShown(true) }, [codeOpen])

  return <div className={`app ${mounted ? 'ready' : ''}`}>
    <Rail />
    <aside className={`dock dock-left ${codeOpen ? 'open' : 'closed'}`} style={{ ['--dock-w' as string]: `${codeWidth}px` }} aria-hidden={!codeOpen}>
      <div className="dock-inner"><div className="dock-content">
        {codeShown && <Suspense fallback={<div className="editor-loading">{tr(locale, 'Loading editor…', '正在加载编辑器…')}</div>}>
          <CodeWorkspace onCollapse={() => useUiStore.getState().togglePanel('codeOpen', false)} collapseIcon={<PanelLeftClose size={15} />} />
        </Suspense>}
      </div></div>
      <ResizeHandle side="left" label={tr(locale, 'Drag to resize · double-click to reset', '拖动调整宽度 · 双击复位')} onDrag={(delta, start) => setWidth('codeWidth', start + delta)} onReset={() => setWidth('codeWidth', defaultWidths.codeWidth)} />
    </aside>
    <main className="stage-slot">
      <ModelScene />
    </main>
    <aside className={`dock dock-right ${sideOpen ? 'open' : 'closed'}`} style={{ ['--dock-w' as string]: `${sideWidth}px` }} aria-hidden={!sideOpen}>
      <ResizeHandle side="right" label={tr(locale, 'Drag to resize · double-click to reset', '拖动调整宽度 · 双击复位')} onDrag={(delta, start) => setWidth('sideWidth', start - delta)} onReset={() => setWidth('sideWidth', defaultWidths.sideWidth)} />
      <div className="dock-inner"><div className="dock-content"><SideDock /></div></div>
    </aside>
    {libraryOpen && <ModelLibrary onClose={() => setLibraryOpen(false)} onImport={() => { setLibraryOpen(false); setOverlay('import') }} />}
    {overlay === 'import' && <ImportModelDialog onClose={closeOverlay} />}
    {overlay === 'palette' && <CommandPalette onClose={closeOverlay} />}
    {overlay === 'shortcuts' && <ShortcutsOverlay onClose={closeOverlay} />}
  </div>
}


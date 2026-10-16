import { useState } from 'react'
import { Check, Command, Download, FileCode2, Keyboard, Layers3, LayoutGrid, Link2, ListTree, Moon, Sun, Upload } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { tr } from '../scene/useViewGraph'
import { createShareLink, exportProject } from './projectActions'

/** The Model Atlas mark (docs/brand/logo.svg): three layers narrowing upward, joined by a tensor path to its output. */
export function BrandMark({ size = 26 }: { size?: number }) {
  const top = { stroke: '#ffffff', strokeOpacity: 0.55, strokeWidth: 1, strokeLinejoin: 'round' as const }
  return <svg width={size} height={size} viewBox="4 2 120 122" aria-hidden="true">
    <defs>
      <radialGradient id="brand-glow"><stop offset="0" stopColor="#ffffff" /><stop offset=".45" stopColor="#f5d0fe" /><stop offset="1" stopColor="#e879f9" stopOpacity="0" /></radialGradient>
    </defs>
    <polygon points="18,86.56 64,113.12 64,121.12 18,94.56" fill="#0ea5e9" /><polygon points="64,113.12 110,86.56 110,94.56 64,121.12" fill="#0369a1" /><polygon points="64,60 110,86.56 64,113.12 18,86.56" fill="#7dd3fc" {...top} />
    <polygon points="29,63.21 64,83.41 64,91.41 29,71.21" fill="#6366f1" /><polygon points="64,83.41 99,63.21 99,71.21 64,91.41" fill="#4338ca" /><polygon points="64,43 99,63.21 64,83.41 29,63.21" fill="#a5b4fc" {...top} />
    <polygon points="40,40.86 64,54.71 64,62.71 40,48.86" fill="#d946ef" /><polygon points="64,54.71 88,40.86 88,48.86 64,62.71" fill="#a21caf" /><polygon points="64,27 88,40.86 64,54.71 40,40.86" fill="#f0abfc" {...top} />
    <line x1="64" y1="40.86" x2="64" y2="12" stroke="#e879f9" strokeWidth="3" strokeLinecap="round" />
    <circle cx="64" cy="12" r="8.5" fill="url(#brand-glow)" />
    <circle cx="64" cy="12" r="4.6" fill="#ffffff" stroke="#d946ef" strokeWidth="2.2" />
    <circle cx="64" cy="40.86" r="3.2" fill="#ffffff" stroke="#d946ef" strokeWidth="1.6" />
  </svg>
}

/** Open or close one section of the right dock, opening the dock itself when needed. */
export function toggleSideSection(section: 'treeOpen' | 'inspectorOpen') {
  const ui = useUiStore.getState()
  const other = section === 'treeOpen' ? 'inspectorOpen' : 'treeOpen'
  if (!ui.sideOpen) { useUiStore.setState({ sideOpen: true, [section]: true, focusMemory: null }); return }
  if (!ui[section]) { ui.togglePanel(section, true); return }
  if (!ui[other]) { useUiStore.setState({ sideOpen: false, [other]: true, focusMemory: null }); return }
  ui.togglePanel(section, false)
}

export function toggleSideDock() {
  const ui = useUiStore.getState()
  if (!ui.sideOpen && !ui.treeOpen && !ui.inspectorOpen) useUiStore.setState({ treeOpen: true, inspectorOpen: true })
  ui.togglePanel('sideOpen')
}

function RailButton({ tip, active, onClick, children, shortcut }: { tip: string; active?: boolean; onClick: () => void; children: React.ReactNode; shortcut?: string }) {
  return <button type="button" className={`rail-button ${active ? 'active' : ''}`} onClick={onClick} data-tip={shortcut ? `${tip}  ${shortcut}` : tip} data-tip-side="right" aria-label={tip} aria-pressed={active}>
    {children}
  </button>
}

/** The slim vertical bar on the far left: panel toggles, project actions and preferences. */
export function Rail() {
  const locale = useModelStore((state) => state.locale)
  const setLocale = useModelStore((state) => state.setLocale)
  const setLibraryOpen = useModelStore((state) => state.setLibraryOpen)
  const codeOpen = useUiStore((state) => state.codeOpen)
  const sideOpen = useUiStore((state) => state.sideOpen)
  const treeOpen = useUiStore((state) => state.treeOpen)
  const inspectorOpen = useUiStore((state) => state.inspectorOpen)
  const resolved = useUiStore((state) => state.resolved)
  const ui = useUiStore.getState
  const [shared, setShared] = useState<'idle' | 'copied' | 'unavailable'>('idle')
  const share = async () => {
    const url = await createShareLink()
    setShared(url ? 'copied' : 'unavailable')
    setTimeout(() => setShared('idle'), 2400)
  }
  const shareTip = shared === 'copied' ? tr(locale, 'Link copied — it reopens this exact view with your edits and notes', '链接已复制：打开即可看到同样的视图、改动和笔记')
    : shared === 'unavailable' ? tr(locale, 'Imported JSON models are shared by exporting the file', '导入的 JSON 模型请用“导出”分享文件')
    : tr(locale, 'Copy share link (code edits, selection, notes)', '复制分享链接（含代码改动、选中模块和笔记）')

  return <nav className="rail" aria-label={tr(locale, 'Workspace', '工作区')}>
    <div className="rail-brand" title="Model Atlas"><BrandMark /></div>
    <div className="rail-group">
      <RailButton tip={tr(locale, 'Code', '代码')} shortcut="[" active={codeOpen} onClick={() => ui().togglePanel('codeOpen')}><FileCode2 size={18} /></RailButton>
      <RailButton tip={tr(locale, 'Model tree', '模型树')} shortcut="]" active={sideOpen && treeOpen} onClick={() => toggleSideSection('treeOpen')}><ListTree size={18} /></RailButton>
      <RailButton tip={tr(locale, 'Module inspector', '模块检查器')} active={sideOpen && inspectorOpen} onClick={() => toggleSideSection('inspectorOpen')}><Layers3 size={18} /></RailButton>
    </div>
    <span className="rail-divider" />
    <div className="rail-group">
      <RailButton tip={tr(locale, 'Model library', '模型库')} onClick={() => setLibraryOpen(true)}><LayoutGrid size={18} /></RailButton>
      <RailButton tip={tr(locale, 'Import .py / JSON', '导入 .py / JSON')} onClick={() => ui().setOverlay('import')}><Upload size={18} /></RailButton>
      <RailButton tip={tr(locale, 'Export project JSON', '导出项目 JSON')} onClick={exportProject}><Download size={18} /></RailButton>
      <RailButton tip={shareTip} active={shared === 'copied'} onClick={() => void share()}>{shared === 'copied' ? <Check size={18} /> : <Link2 size={18} />}</RailButton>
    </div>
    <div className="rail-spacer" />
    <div className="rail-group">
      <RailButton tip={tr(locale, 'Command palette', '命令面板')} shortcut="Ctrl K" onClick={() => ui().setOverlay('palette')}><Command size={17} /></RailButton>
      <RailButton tip={tr(locale, 'Keyboard shortcuts', '快捷键')} shortcut="?" onClick={() => ui().setOverlay('shortcuts')}><Keyboard size={18} /></RailButton>
      <RailButton tip={resolved === 'dark' ? tr(locale, 'Light mode', '切换明亮模式') : tr(locale, 'Dark mode', '切换黑暗模式')} shortcut="T" onClick={() => ui().cycleTheme()}>
        <span className="theme-icon" data-mode={resolved}><Sun size={18} /><Moon size={18} /></span>
      </RailButton>
      <RailButton tip={locale === 'zh' ? 'Switch to English' : '切换到中文'} onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}><span className="rail-text">{locale === 'zh' ? '中' : 'EN'}</span></RailButton>
    </div>
  </nav>
}

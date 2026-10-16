import { useState } from 'react'
import { Check, Command, Download, FileCode2, Keyboard, Layers3, LayoutGrid, Link2, ListTree, Moon, Sun, Upload } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { tr } from '../scene/useViewGraph'
import { createShareLink, exportProject } from './projectActions'

export function BrandMark({ size = 26 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
    <defs>
      <linearGradient id="brand-gradient" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#38bdf8" /><stop offset="0.55" stopColor="#818cf8" /><stop offset="1" stopColor="#e879f9" /></linearGradient>
    </defs>
    <path d="M16 2.5 L28.5 9.6 L28.5 22.4 L16 29.5 L3.5 22.4 L3.5 9.6 Z" fill="url(#brand-gradient)" opacity="0.16" />
    <path d="M16 2.5 L28.5 9.6 L28.5 22.4 L16 29.5 L3.5 22.4 L3.5 9.6 Z" fill="none" stroke="url(#brand-gradient)" strokeWidth="1.9" strokeLinejoin="round" />
    <path d="M3.5 9.6 L16 16.8 L28.5 9.6 M16 16.8 L16 29.5" fill="none" stroke="url(#brand-gradient)" strokeWidth="1.5" strokeLinejoin="round" opacity="0.75" />
    <circle cx="16" cy="16.8" r="2.7" fill="url(#brand-gradient)" />
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

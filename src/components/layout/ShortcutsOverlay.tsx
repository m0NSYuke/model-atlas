import { useEffect } from 'react'
import { Keyboard, X } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { tr } from '../scene/useViewGraph'

const groups: { title: [string, string]; rows: [string[], string, string][] }[] = [
  {
    title: ['Panels', '面板'],
    rows: [
      [['Ctrl', 'K'], 'Command palette', '命令面板'],
      [['['], 'Show / hide code', '显示 / 收起代码栏'],
      [['Ctrl', 'B'], 'Show / hide code (in the editor too)', '显示 / 收起代码栏（编辑器内也可用）'],
      [[']'], 'Show / hide tree and inspector', '显示 / 收起模型树与检查器'],
      [['\\'], 'Focus mode: hide every panel', '专注模式：收起所有侧栏'],
      [['T'], 'Light / dark theme', '切换明亮 / 黑暗模式'],
    ],
  },
  {
    title: ['View', '视图'],
    rows: [
      [['1'], '3D view', '3D 视图'],
      [['2'], 'Flat diagram', '平面结构图'],
      [['F'], 'Fit the model in view', '适配视野'],
      [['V'], 'Angled / front camera', '斜视 / 正视'],
      [['L'], 'Cycle labels', '切换标签显示'],
      [['R'], 'Trace connections of the selection', '追踪选中节点的连接'],
      [['P'], 'Tensor flow particles', '张量流动粒子'],
      [['Esc'], 'Clear selection', '取消选中'],
    ],
  },
  {
    title: ['Mouse', '鼠标'],
    rows: [
      [['Drag'], 'Orbit', '旋转视角'],
      [['Right-drag'], 'Pan', '平移'],
      [['Wheel'], 'Zoom', '缩放'],
      [['Double-click'], 'Open a module / collapse its frame', '展开模块 / 双击框标签收起'],
    ],
  },
]

export function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  const locale = useModelStore((state) => state.locale)
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') { event.preventDefault(); onClose() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  const mouse = (key: string) => locale === 'zh' ? ({ Drag: '左键拖动', 'Right-drag': '右键拖动', Wheel: '滚轮', 'Double-click': '双击' } as Record<string, string>)[key] ?? key : key
  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="shortcuts-dialog" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title">
      <header className="dialog-heading">
        <div className="dialog-icon"><Keyboard size={18} /></div>
        <div><h2 id="shortcuts-title">{tr(locale, 'Keyboard shortcuts', '快捷键')}</h2><p>{tr(locale, 'Single-key shortcuts pause while you type in the editor.', '在编辑器里输入时，单键快捷键会自动暂停。')}</p></div>
        <button type="button" className="icon-button" aria-label={tr(locale, 'Close', '关闭')} onClick={onClose}><X size={18} /></button>
      </header>
      <div className="shortcuts-grid">
        {groups.map((group) => <section key={group.title[0]}>
          <h3>{tr(locale, ...group.title)}</h3>
          {group.rows.map(([keys, en, zh]) => <div className="shortcut-row" key={en}>
            <span>{tr(locale, en, zh)}</span>
            <span className="keys">{keys.map((key) => <kbd key={key}>{mouse(key)}</kbd>)}</span>
          </div>)}
        </section>)}
      </div>
    </div>
  </div>
}

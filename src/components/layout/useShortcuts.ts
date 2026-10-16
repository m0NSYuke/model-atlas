import { useEffect } from 'react'
import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { toggleSideDock } from './Rail'

function typing(target: EventTarget | null): boolean {
  const element = target as HTMLElement | null
  if (!element) return false
  return element.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(element.tagName) || Boolean(element.closest?.('.monaco-editor'))
}

/** Global keyboard shortcuts. Single keys are ignored while typing; Ctrl/⌘ combos work everywhere. */
export function useShortcuts() {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const ui = useUiStore.getState()
      const model = useModelStore.getState()
      const mod = event.ctrlKey || event.metaKey
      const key = event.key.toLowerCase()
      if (mod && !event.shiftKey && !event.altKey && key === 'k') {
        event.preventDefault()
        event.stopPropagation()
        ui.setOverlay(ui.overlay === 'palette' ? null : 'palette')
        return
      }
      if (mod && !event.shiftKey && !event.altKey && key === 'b') {
        event.preventDefault()
        event.stopPropagation()
        ui.togglePanel('codeOpen')
        return
      }
      if (event.key === 'Escape') {
        if (ui.overlay || model.libraryOpen) return // dialogs handle their own Escape
        if (typing(event.target)) return
        if (model.selectedNodeId) model.resetSelection()
        else if (ui.focusMemory) ui.toggleFocusMode(false)
        return
      }
      if (event.key === '?' && ui.overlay === 'shortcuts') { event.preventDefault(); ui.setOverlay(null); return }
      if (mod || event.altKey || typing(event.target) || ui.overlay || model.libraryOpen) return
      const actions: Record<string, () => void> = {
        '[': () => ui.togglePanel('codeOpen'),
        ']': () => toggleSideDock(),
        '\\': () => ui.toggleFocusMode(),
        '1': () => ui.setView('3d'),
        '2': () => ui.setView('2d'),
        f: () => ui.fit(),
        l: () => ui.cycleLabels(),
        r: () => ui.setTrace(!ui.trace),
        p: () => ui.setFlow(!ui.flow),
        v: () => ui.setAngle(ui.angle === 'iso' ? 'front' : 'iso'),
        t: () => ui.cycleTheme(),
        '?': () => ui.setOverlay('shortcuts'),
        '/': () => ui.setOverlay('palette'),
      }
      const action = actions[event.key] ?? actions[key]
      if (action) { event.preventDefault(); action() }
    }
    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [])
}

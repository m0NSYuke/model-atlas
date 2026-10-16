import { create } from 'zustand'

export type ThemeSetting = 'light' | 'dark' | 'system'
export type ResolvedTheme = 'light' | 'dark'
export type ViewMode = '3d' | '2d'
export type LabelMode = 'smart' | 'all' | 'none'
export type CameraAngle = 'iso' | 'front'
export type Overlay = null | 'palette' | 'shortcuts' | 'import'

const storageKey = 'atlas-ui-v1'

interface SavedUi {
  theme: ThemeSetting
  codeOpen: boolean
  sideOpen: boolean
  treeOpen: boolean
  inspectorOpen: boolean
  codeWidth: number
  sideWidth: number
  split: number
  labels: LabelMode
  trace: boolean
  flow: boolean
  view: ViewMode
}

interface UiState extends SavedUi {
  resolved: ResolvedTheme
  angle: CameraAngle
  fitRequest: number
  overlay: Overlay
  /** Panels hidden by focus mode, restored when it ends. */
  focusMemory: { codeOpen: boolean; sideOpen: boolean } | null
  setTheme: (theme: ThemeSetting) => void
  cycleTheme: () => void
  togglePanel: (panel: 'codeOpen' | 'sideOpen' | 'treeOpen' | 'inspectorOpen', open?: boolean) => void
  setWidth: (panel: 'codeWidth' | 'sideWidth', width: number) => void
  setSplit: (split: number) => void
  setView: (view: ViewMode) => void
  setAngle: (angle: CameraAngle) => void
  cycleLabels: () => void
  setTrace: (trace: boolean) => void
  setFlow: (flow: boolean) => void
  fit: () => void
  toggleFocusMode: (on?: boolean) => void
  setOverlay: (overlay: Overlay) => void
}

// First-run dock widths scale with the window so the 3D stage keeps at least half of a laptop screen.
const screenWidth = typeof window !== 'undefined' ? window.innerWidth : 1600
const defaults: SavedUi = {
  theme: 'dark', codeOpen: true, sideOpen: true, treeOpen: true, inspectorOpen: true,
  codeWidth: Math.round(Math.min(440, Math.max(340, screenWidth * 0.27))), sideWidth: Math.round(Math.min(340, Math.max(290, screenWidth * 0.21))),
  split: 0.42, labels: 'smart', trace: true, flow: true, view: '3d',
}
export const defaultWidths = { codeWidth: defaults.codeWidth, sideWidth: defaults.sideWidth }

export const panelLimits = { codeWidth: [320, 820] as const, sideWidth: [280, 540] as const }
const clampSplit = (split: number) => Math.min(0.82, Math.max(0.16, split))
const clampWidth = (panel: 'codeWidth' | 'sideWidth', width: number) => Math.round(Math.min(panelLimits[panel][1], Math.max(panelLimits[panel][0], width)))

const isBoolean = (value: unknown) => typeof value === 'boolean'
const isNumber = (value: unknown) => typeof value === 'number' && Number.isFinite(value)
const oneOf = (...values: string[]) => (value: unknown) => values.includes(value as string)
const savedChecks: Record<keyof SavedUi, (value: unknown) => boolean> = {
  theme: oneOf('light', 'dark', 'system'), codeOpen: isBoolean, sideOpen: isBoolean, treeOpen: isBoolean, inspectorOpen: isBoolean,
  codeWidth: isNumber, sideWidth: isNumber, split: isNumber, labels: oneOf('smart', 'all', 'none'), trace: isBoolean, flow: isBoolean, view: oneOf('3d', '2d'),
}

/** Saved settings, field by field: anything missing, of the wrong type or out of range falls back to the default. */
function load(): SavedUi {
  let saved: unknown = null
  try { saved = JSON.parse(window.localStorage.getItem(storageKey) ?? 'null') } catch { /* storage unavailable or corrupt */ }
  const result: SavedUi = { ...defaults }
  if (saved && typeof saved === 'object') {
    for (const key of Object.keys(savedChecks) as (keyof SavedUi)[]) {
      const value = (saved as Record<string, unknown>)[key]
      if (savedChecks[key](value)) Object.assign(result, { [key]: value })
    }
  }
  return { ...result, codeWidth: clampWidth('codeWidth', result.codeWidth), sideWidth: clampWidth('sideWidth', result.sideWidth), split: clampSplit(result.split) }
}

/** Below this width the docks float over the stage (see `.dock` in styles/index.css) instead of sitting beside it. */
export const overlayBelow = 900
const railWidth = 56
const gap = 8

/** Width the open docks may share so the stage keeps at least 40% of the window (and never less than 360px). */
function dockBudget(windowWidth: number, open: number): number {
  return windowWidth - railWidth - gap * (1 + open) - Math.max(360, Math.round(windowWidth * 0.4))
}

/**
 * The widths the docks are drawn at: the saved widths, shrunk towards their minimums when both together would squeeze the
 * stage. The saved widths are kept, so the docks grow back when the window widens.
 */
export function fitDocks(state: Pick<SavedUi, 'codeOpen' | 'sideOpen' | 'codeWidth' | 'sideWidth'>, windowWidth: number): { codeWidth: number; sideWidth: number } {
  const { codeWidth, sideWidth } = state
  if (windowWidth < overlayBelow) return { codeWidth, sideWidth }
  const budget = dockBudget(windowWidth, Number(state.codeOpen) + Number(state.sideOpen))
  const code = state.codeOpen ? codeWidth : 0
  const side = state.sideOpen ? sideWidth : 0
  const excess = code + side - budget
  if (excess <= 0) return { codeWidth, sideWidth }
  // Take the excess from each dock in proportion to how far it is above its minimum.
  const slack = { codeWidth: state.codeOpen ? Math.max(0, codeWidth - panelLimits.codeWidth[0]) : 0, sideWidth: state.sideOpen ? Math.max(0, sideWidth - panelLimits.sideWidth[0]) : 0 }
  const total = slack.codeWidth + slack.sideWidth
  if (!total) return { codeWidth, sideWidth }
  const cut = Math.min(excess, total)
  return { codeWidth: Math.round(codeWidth - cut * slack.codeWidth / total), sideWidth: Math.round(sideWidth - cut * slack.sideWidth / total) }
}

const media = typeof window !== 'undefined' && window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null
const resolve = (theme: ThemeSetting): ResolvedTheme => theme === 'system' ? (media?.matches ? 'dark' : 'light') : theme

function applyTheme(theme: ResolvedTheme) {
  const root = document.documentElement
  root.dataset.theme = theme
  root.style.colorScheme = theme
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#0a0c12' : '#f4f5f9')
}

const initial = load()
applyTheme(resolve(initial.theme))

export const useUiStore = create<UiState>((set, get) => ({
  ...initial,
  resolved: resolve(initial.theme),
  angle: 'iso',
  fitRequest: 0,
  overlay: null,
  focusMemory: null,
  setTheme: (theme) => {
    const resolved = resolve(theme)
    applyTheme(resolved)
    set({ theme, resolved })
  },
  cycleTheme: () => get().setTheme(get().resolved === 'dark' ? 'light' : 'dark'),
  togglePanel: (panel, open) => set((state) => ({ [panel]: open ?? !state[panel], focusMemory: null })),
  setWidth: (panel, width) => {
    const state = get()
    const other = panel === 'codeWidth' ? 'sideWidth' : 'codeWidth'
    const windowWidth = window.innerWidth
    if (windowWidth < overlayBelow) { set({ [panel]: clampWidth(panel, width) }); return }
    // The other dock keeps the width it is drawn at, and this one may only take what is left beside the stage.
    const fitted = fitDocks(state, windowWidth)
    const otherOpen = other === 'codeWidth' ? state.codeOpen : state.sideOpen
    const room = otherOpen ? dockBudget(windowWidth, 2) - fitted[other] : dockBudget(windowWidth, 1)
    set({ [panel]: clampWidth(panel, Math.min(width, room)), ...(otherOpen ? { [other]: fitted[other] } : {}) })
  },
  setSplit: (split) => set({ split: clampSplit(split) }),
  setView: (view) => set({ view }),
  setAngle: (angle) => set({ angle }),
  cycleLabels: () => set((state) => ({ labels: state.labels === 'smart' ? 'all' : state.labels === 'all' ? 'none' : 'smart' })),
  setTrace: (trace) => set({ trace }),
  setFlow: (flow) => set({ flow }),
  fit: () => set((state) => ({ fitRequest: state.fitRequest + 1 })),
  toggleFocusMode: (on) => {
    const state = get()
    const active = state.focusMemory !== null || (!state.codeOpen && !state.sideOpen)
    const next = on ?? !active
    if (next && state.focusMemory === null) set({ focusMemory: { codeOpen: state.codeOpen, sideOpen: state.sideOpen }, codeOpen: false, sideOpen: false })
    if (!next) {
      const memory = state.focusMemory ?? { codeOpen: true, sideOpen: true }
      set({ ...memory, focusMemory: null })
    }
  },
  setOverlay: (overlay) => set({ overlay }),
}))

media?.addEventListener('change', () => {
  const { theme, setTheme } = useUiStore.getState()
  if (theme === 'system') setTheme('system')
})

/** True while the side dock is folded away only because the window is of medium width. */
let sideAutoClosed = false
let saveTimer: ReturnType<typeof setTimeout> | undefined
useUiStore.subscribe((state) => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    const { theme, codeOpen, sideOpen, treeOpen, inspectorOpen, codeWidth, sideWidth, split, labels, trace, flow, view, focusMemory } = state
    // Focus mode and the side dock folded away on a medium window are temporary: persist the panels the user chose.
    const panels = focusMemory ?? { codeOpen, sideOpen }
    const saved: SavedUi = { theme, ...panels, sideOpen: panels.sideOpen || sideAutoClosed, treeOpen, inspectorOpen, codeWidth, sideWidth, split, labels, trace, flow, view }
    try { window.localStorage.setItem(storageKey, JSON.stringify(saved)) } catch { /* storage may be unavailable */ }
  }, 250)
})

// On a narrow window (a phone, or a small split screen) the code and side panels would cover the whole 3D view,
// so they start hidden, like focus mode; widening the window brings back the panels that were open.
// On a medium window only one dock fits beside the stage, so the side dock starts folded away and comes back on a wide one.
if (typeof window !== 'undefined' && window.matchMedia) {
  const narrow = window.matchMedia(`(max-width: ${overlayBelow - 1}px)`)
  const medium = window.matchMedia(`(min-width: ${overlayBelow}px) and (max-width: 1100px)`)
  let autoHidden = false
  const apply = () => {
    const state = useUiStore.getState()
    if (narrow.matches && (state.codeOpen || state.sideOpen) && state.focusMemory === null) { state.toggleFocusMode(true); autoHidden = true }
    else if (!narrow.matches && autoHidden) { if (useUiStore.getState().focusMemory !== null) state.toggleFocusMode(false); autoHidden = false }
    const current = useUiStore.getState()
    if (medium.matches && current.codeOpen && current.sideOpen && current.focusMemory === null) { useUiStore.setState({ sideOpen: false }); sideAutoClosed = true }
    else if (!medium.matches && !narrow.matches && sideAutoClosed) { if (!current.sideOpen && current.focusMemory === null) useUiStore.setState({ sideOpen: true }); sideAutoClosed = false }
  }
  apply()
  narrow.addEventListener('change', apply)
  medium.addEventListener('change', apply)
  // Opening the side dock by hand ends the automatic folding.
  useUiStore.subscribe((state, previous) => { if (state.sideOpen && !previous.sideOpen) sideAutoClosed = false })
}

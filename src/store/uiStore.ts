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
}

interface UiState extends SavedUi {
  resolved: ResolvedTheme
  view: ViewMode
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
  split: 0.42, labels: 'smart', trace: true, flow: true,
}
export const defaultWidths = { codeWidth: defaults.codeWidth, sideWidth: defaults.sideWidth }

export const panelLimits = { codeWidth: [320, 820] as const, sideWidth: [280, 540] as const }

function load(): SavedUi {
  try {
    const saved = JSON.parse(window.localStorage.getItem(storageKey) ?? 'null') as Partial<SavedUi> | null
    return { ...defaults, ...(saved ?? {}) }
  } catch { return defaults }
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
  view: '3d',
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
    const [low, high] = panelLimits[panel]
    set({ [panel]: Math.round(Math.min(high, Math.max(low, width))) })
  },
  setSplit: (split) => set({ split: Math.min(0.82, Math.max(0.16, split)) }),
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

let saveTimer: ReturnType<typeof setTimeout> | undefined
useUiStore.subscribe((state) => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    const { theme, codeOpen, sideOpen, treeOpen, inspectorOpen, codeWidth, sideWidth, split, labels, trace, flow, focusMemory } = state
    // Focus mode is temporary: persist the panels the user had before entering it.
    const panels = focusMemory ?? { codeOpen, sideOpen }
    const saved: SavedUi = { theme, ...panels, treeOpen, inspectorOpen, codeWidth, sideWidth, split, labels, trace, flow }
    try { window.localStorage.setItem(storageKey, JSON.stringify(saved)) } catch { /* storage may be unavailable */ }
  }, 250)
})

// On a narrow window (a phone, or a small split screen) the code and side panels would cover the whole 3D view,
// so they start hidden, like focus mode; widening the window brings back the panels that were open.
if (typeof window !== 'undefined' && window.matchMedia) {
  const narrow = window.matchMedia('(max-width: 899px)')
  let autoHidden = false
  const apply = () => {
    const state = useUiStore.getState()
    if (narrow.matches && (state.codeOpen || state.sideOpen) && state.focusMemory === null) { state.toggleFocusMode(true); autoHidden = true }
    else if (!narrow.matches && autoHidden) { if (useUiStore.getState().focusMemory !== null) state.toggleFocusMode(false); autoHidden = false }
  }
  apply()
  narrow.addEventListener('change', apply)
}

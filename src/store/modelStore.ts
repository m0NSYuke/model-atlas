import { create } from 'zustand'
import type { Locale, ModelIR } from '../core/model-ir/types'
import { importModel, type AtlasProject } from '../core/model-ir/importModel'
import { updateVaeShapes } from '../core/shape-engine/inferShape'
import { buildModelFromCode, CodeBuildError, hasPythonModel, type CodeDiagnostic } from '../core/pytorch'
import { autoDepth, depthCollapse, groupIds } from '../core/layout/viewGraph'
import { codeFiles as demoCodeFiles } from '../data/models/vae3d/codeFiles'
import { studyById } from '../data/models/studyCatalog'
import { blankTemplate, codeModelById } from '../data/models/codeCatalog'
import { builtInProjectKey, listProjects, readProject, removeProject, sameFiles, uniqueProjectKey, writeProject, type ProjectInfo } from './savedProjects'
import { codeFilesOrNull } from '../core/validate'

const storageKey = 'model-atlas-project-v2'
const tr = (locale: Locale, en: string, zh: string) => locale === 'zh' ? zh : en

export type ModelOrigin = 'demo' | 'catalog' | 'import' | 'code'

interface SavedProject {
  origin: ModelOrigin
  builtInId: string | null
  locale: Locale
  model: ModelIR
  codeFiles: Record<string, string>
  /** Which entry of "My projects" this code is saved under. */
  projectKey?: string | null
  projectName?: string | null
}

interface BuiltProject {
  model: ModelIR
  diagnostic: CodeDiagnostic | null
  logs: string[]
}

/** Trace Python source; on failure keep the previous model and report where the code broke. */
function trace(id: string, files: Record<string, string>, fallback: ModelIR | null, meta?: Parameters<typeof buildModelFromCode>[2]): BuiltProject {
  try {
    const { model, logs } = buildModelFromCode(id, files, meta)
    return { model, diagnostic: null, logs }
  } catch (error) {
    const diagnostic = error instanceof CodeBuildError ? error.diagnostic : { message: error instanceof Error ? error.message : String(error) }
    return { model: fallback ?? emptyModel(id), diagnostic, logs: [] }
  }
}

function emptyModel(id: string): ModelIR {
  return { id, name: 'Model', layout: 'code', nodes: [{ id: 'empty', name: 'No traced graph yet', type: 'generic' }], edges: [] }
}

function metaFor(id: string | null): Parameters<typeof buildModelFromCode>[2] {
  const entry = id ? codeModelById[id] : undefined
  if (!entry) return undefined
  return { name: entry.name, description: entry.description, localized: { zh: { name: entry.zhName, description: entry.zhDescription } } }
}

function initialProject(): SavedProject & { diagnostic: CodeDiagnostic | null; logs: string[] } {
  if (typeof window !== 'undefined') {
    try {
      const saved = JSON.parse(window.localStorage.getItem(storageKey) ?? 'null') as SavedProject | null
      if (saved && ['demo', 'catalog', 'import', 'code'].includes(saved.origin)) {
        const locale: Locale = saved.locale === 'en' ? 'en' : 'zh'
        if (saved.origin === 'code') {
          const codeFiles = codeFilesOrNull(saved.codeFiles)
          if (!codeFiles) throw new Error('Saved project has no code files')
          const builtInId = typeof saved.builtInId === 'string' ? saved.builtInId : null
          const savedId = typeof saved.model?.id === 'string' ? saved.model.id : null
          const pristine = builtInId && codeModelById[builtInId] ? trace(builtInId, codeModelById[builtInId].files, null, metaFor(builtInId)).model : null
          const built = trace(savedId ?? builtInId ?? 'my-model', codeFiles, pristine, metaFor(builtInId))
          const projectKey = typeof saved.projectKey === 'string' ? saved.projectKey : (builtInId ? builtInProjectKey(builtInId) : uniqueProjectKey(savedId ?? 'my-model'))
          const projectName = typeof saved.projectName === 'string' ? saved.projectName : undefined
          return { ...saved, builtInId, codeFiles, projectName, locale, model: built.model, diagnostic: built.diagnostic, logs: built.logs, projectKey }
        }
        const validated = importModel({ model: saved.model, codeFiles: saved.codeFiles })
        const current = studyById[validated.model.id]?.model
        const model = saved.origin === 'demo' && current ? updateVaeShapes(current, validated.codeFiles)
          : saved.origin === 'catalog' && current ? current : validated.model
        return { origin: saved.origin, builtInId: saved.builtInId ?? null, locale, model, codeFiles: validated.codeFiles, diagnostic: null, logs: [] }
      }
    } catch { /* Ignore stale or invalid local data. */ }
  }
  const entry = codeModelById['code-gpt']
  const built = trace(entry.id, entry.files, null, metaFor(entry.id))
  return { origin: 'code', builtInId: entry.id, locale: 'zh', model: built.model, codeFiles: { ...entry.files }, diagnostic: built.diagnostic, logs: built.logs, projectKey: builtInProjectKey(entry.id) }
}

const initial = initialProject()
let parseTimer: ReturnType<typeof setTimeout> | undefined
let saveTimer: ReturnType<typeof setTimeout> | undefined

/** Nodes that are new or whose shapes/parameters changed between two traces. */
function changedNodes(previous: ModelIR, next: ModelIR): string[] {
  const before = new Map(previous.nodes.map((node) => [node.id, node]))
  return next.nodes.filter((node) => {
    const old = before.get(node.id)
    return !old || JSON.stringify(old.outputShape) !== JSON.stringify(node.outputShape) || old.paramCount !== node.paramCount || JSON.stringify(old.parameters) !== JSON.stringify(node.parameters)
  }).map((node) => node.id)
}

function firstSelection(origin: ModelOrigin): string | null {
  return origin === 'demo' ? 'downsample' : null
}

interface ModelState {
  modelIR: ModelIR
  modelOrigin: ModelOrigin
  builtInId: string | null
  locale: Locale
  selectedNodeId: string | null
  hoveredNodeId: string | null
  activeFile: string
  activeLine: number | null
  sourceRevision: number
  focusRevision: number
  codeFiles: Record<string, string>
  parseError: string | null
  diagnostic: CodeDiagnostic | null
  logs: string[]
  /** null = choose an expansion depth automatically. */
  depth: number | null
  toggles: Record<string, boolean>
  changed: { ids: string[]; stamp: number }
  libraryOpen: boolean
  /** The "My projects" entry the current code is saved to (null for study maps and Model IR imports). */
  projectKey: string | null
  /** The project's name in "My projects" (built-in models use their own name). */
  projectName: string | null
  projects: ProjectInfo[]
  /** The browser refused the last save (storage full or disabled). */
  saveFailed: boolean
  selectNode: (id: string, options?: { fromEditor?: boolean }) => void
  hoverNode: (id: string | null) => void
  resetSelection: () => void
  setLocale: (locale: Locale) => void
  setActiveFile: (file: string) => void
  openSource: (file: string, line: number) => void
  updateCode: (file: string, value: string) => void
  applyEdit: (file: string, find: string, replace: string) => boolean
  resetCode: () => void
  importProject: (project: AtlasProject) => void
  importPython: (fileName: string, source: string) => void
  /** A local project: chosen .py files plus an optional atlas.json. */
  importFiles: (name: string, files: Record<string, string>) => void
  loadBuiltIn: (id: string) => void
  /** Open code (a shared link): files traced as one model, linked to a built-in model when given. */
  loadCodeFiles: (id: string, files: Record<string, string>, builtInId: string | null) => void
  loadBlank: () => void
  resetDemo: () => void
  updateModelIR: (model: ModelIR) => void
  setDepth: (depth: number | null) => void
  toggleCollapsed: (id: string, collapsed?: boolean) => void
  setLibraryOpen: (open: boolean) => void
  openProject: (key: string) => void
  deleteProject: (key: string) => void
}

export function effectiveDepth(state: Pick<ModelState, 'depth' | 'modelIR'>): number {
  return state.depth ?? autoDepth(state.modelIR)
}

/** Whether a group is drawn as a single collapsed block. */
export function collapsedPredicate(state: Pick<ModelState, 'depth' | 'modelIR' | 'toggles'>): (id: string) => boolean {
  const groups = groupIds(state.modelIR)
  const byDepth = depthCollapse(state.modelIR, effectiveDepth(state))
  return (id: string) => groups.has(id) && (state.toggles[id] ?? byDepth(id))
}

export const useModelStore = create<ModelState>((set, get) => {
  const rebuild = () => {
    const state = get()
    if (state.modelOrigin !== 'code') return
    const built = trace(state.modelIR.id, state.codeFiles, state.modelIR, metaFor(state.builtInId))
    if (built.diagnostic) { set({ diagnostic: built.diagnostic, parseError: built.diagnostic.message }); return }
    const ids = changedNodes(state.modelIR, built.model)
    const keepSelection = state.selectedNodeId && built.model.nodes.some((node) => node.id === state.selectedNodeId)
    set((current) => ({
      modelIR: built.model, diagnostic: null, parseError: null, logs: built.logs,
      selectedNodeId: keepSelection ? current.selectedNodeId : null,
      changed: { ids: ids.length === built.model.nodes.length ? [] : ids, stamp: Date.now() },
    }))
  }

  /** Saves the current code first; asks before switching only when that save failed. */
  const leaveCurrent = (): boolean => {
    clearTimeout(parseTimer)
    if (saveNow()) return true
    return typeof window === 'undefined' || window.confirm(tr(get().locale, 'Your changes could not be saved in this browser (storage is full or disabled). Open another model anyway and lose them? Export the project first to keep them.', '浏览器没能保存你的修改（存储已满或被禁用）。仍然打开别的模型并丢掉这些修改吗？想保留请先导出项目。'))
  }

  /** Where the code is saved: an existing key, or a new project named `fresh`. */
  type Target = string | { fresh: string; name: string }
  const loadCode = (id: string, files: Record<string, string>, builtInId: string | null, target: Target) => {
    if (!leaveCurrent()) return
    const current = get().projectKey
    const projectKey = typeof target === 'string' ? target : uniqueProjectKey(target.fresh, current ? [current] : [])
    const projectName = typeof target === 'string' ? listProjects().find((item) => item.key === target)?.name ?? null : target.name
    const built = trace(id, files, null, metaFor(builtInId))
    set((state) => ({
      modelIR: built.model, modelOrigin: 'code', builtInId, codeFiles: { ...files }, projectKey, projectName,
      selectedNodeId: null, hoveredNodeId: null, depth: (builtInId ? codeModelById[builtInId]?.depth : undefined) ?? null, toggles: {},
      activeFile: 'model.py' in files ? 'model.py' : Object.keys(files).find((file) => file.endsWith('.py')) ?? Object.keys(files)[0] ?? 'model.py', activeLine: null,
      sourceRevision: state.sourceRevision + 1, focusRevision: state.focusRevision + 1,
      diagnostic: built.diagnostic, parseError: built.diagnostic?.message ?? null, logs: built.logs,
      changed: { ids: [], stamp: Date.now() },
    }))
  }

  return {
    modelIR: initial.model,
    modelOrigin: initial.origin,
    builtInId: initial.builtInId,
    locale: initial.locale,
    selectedNodeId: firstSelection(initial.origin),
    hoveredNodeId: null,
    activeFile: initial.origin === 'demo' ? 'blocks.py' : Object.keys(initial.codeFiles)[0] ?? 'model-ir.json',
    activeLine: null,
    sourceRevision: 0,
    focusRevision: 0,
    codeFiles: initial.codeFiles,
    parseError: initial.diagnostic?.message ?? null,
    diagnostic: initial.diagnostic,
    logs: initial.logs,
    depth: null,
    toggles: {},
    changed: { ids: [], stamp: 0 },
    libraryOpen: false,
    projectKey: initial.origin === 'code' ? initial.projectKey ?? null : null,
    projectName: initial.origin === 'code' ? initial.projectName ?? null : null,
    projects: listProjects(),
    saveFailed: false,
    selectNode: (id, options) => {
      const state = get()
      const node = state.modelIR.nodes.find((item) => item.id === id)
      if (!node) return
      // Expand collapsed ancestors so the node becomes visible.
      const toggles = { ...state.toggles }
      const isCollapsed = collapsedPredicate(state)
      const parents = new Map(state.modelIR.nodes.map((item) => [item.id, item.parentId]))
      let changedToggles = false
      for (let current = node.parentId; current; current = parents.get(current)) {
        if (isCollapsed(current)) { toggles[current] = false; changedToggles = true }
      }
      const file = node.source?.file
      const linkSource = !options?.fromEditor && file && file in state.codeFiles
      set((current) => ({
        selectedNodeId: id,
        toggles: changedToggles ? toggles : current.toggles,
        activeFile: linkSource ? file! : current.activeFile,
        activeLine: linkSource ? node.source!.startLine : current.activeLine,
        sourceRevision: options?.fromEditor ? current.sourceRevision : current.sourceRevision + 1,
        focusRevision: current.focusRevision + 1,
      }))
    },
    hoverNode: (id) => { if (get().hoveredNodeId !== id) set({ hoveredNodeId: id }) },
    resetSelection: () => set({ selectedNodeId: null }),
    setLocale: (locale) => set({ locale }),
    setActiveFile: (activeFile) => set({ activeFile, activeLine: null }),
    openSource: (activeFile, activeLine) => set((state) => ({ activeFile, activeLine, sourceRevision: state.sourceRevision + 1 })),
    updateCode: (file, value) => {
      set((state) => ({ codeFiles: { ...state.codeFiles, [file]: value } }))
      const origin = get().modelOrigin
      if (origin === 'code') {
        if (!file.endsWith('.py') && file !== 'atlas.json') return
        clearTimeout(parseTimer)
        parseTimer = setTimeout(rebuild, 350)
        return
      }
      if (origin !== 'demo' || !['model.py', 'blocks.py'].includes(file)) return
      clearTimeout(parseTimer)
      parseTimer = setTimeout(() => {
        try {
          const modelIR = updateVaeShapes(studyById.vae3d.model, get().codeFiles)
          set({ modelIR, parseError: null, diagnostic: null })
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Could not parse parameters'
          set({ parseError: message, diagnostic: { message } })
        }
      }, 400)
    },
    applyEdit: (file, find, replace) => {
      const source = get().codeFiles[file]
      if (source === undefined || !source.includes(find)) return false
      const next = source.replace(find, replace)
      const line = source.slice(0, source.indexOf(find)).split('\n').length
      set((state) => ({ codeFiles: { ...state.codeFiles, [file]: next }, activeFile: file, activeLine: line, sourceRevision: state.sourceRevision + 1 }))
      clearTimeout(parseTimer)
      rebuild()
      return true
    },
    resetCode: () => {
      const { builtInId, modelOrigin } = get()
      // For an edited built-in model the saved copy is dropped once the original code is back (see saveNow).
      if (modelOrigin === 'code' && builtInId && codeModelById[builtInId]) loadCode(builtInId, codeModelById[builtInId].files, builtInId, get().projectKey ?? builtInProjectKey(builtInId))
      else if (modelOrigin === 'demo') get().loadBuiltIn('vae3d')
    },
    importProject: ({ model, codeFiles }) => {
      clearTimeout(parseTimer)
      if (hasPythonModel(codeFiles)) {
        const built = trace(model.id, codeFiles, null)
        if (!built.diagnostic) { loadCode(model.id, codeFiles, null, { fresh: model.id, name: model.name || model.id }); return }
      }
      if (!leaveCurrent()) return
      set((state) => ({
        modelIR: model, modelOrigin: 'import', builtInId: null, codeFiles, projectKey: null, projectName: null,
        selectedNodeId: model.nodes[0]?.id ?? null, hoveredNodeId: null, depth: null, toggles: {},
        activeFile: Object.keys(codeFiles)[0] ?? 'model-ir.json', activeLine: null,
        sourceRevision: state.sourceRevision + 1, parseError: null, diagnostic: null, logs: [],
      }))
    },
    importPython: (fileName, source) => {
      const id = fileName.replace(/\.py$/, '').replace(/[^\w-]/g, '-') || 'my-model'
      loadCode(id, { [fileName.endsWith('.py') ? fileName : 'model.py']: source }, null, { fresh: id, name: fileName })
    },
    importFiles: (name, files) => {
      const id = name.toLowerCase().replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '') || 'my-model'
      loadCode(id, files, null, { fresh: id, name })
    },
    loadBuiltIn: (id) => {
      if (codeModelById[id]) {
        // A built-in model you edited before opens with your edits; Reset brings back the original.
        const key = builtInProjectKey(id)
        if (get().projectKey === key) return
        const saved = readProject(key)
        loadCode(id, saved?.codeFiles ?? codeModelById[id].files, id, key)
        return
      }
      const entry = studyById[id]
      if (!entry) return
      if (!leaveCurrent()) return
      const isDemo = id === 'vae3d'
      set((state) => ({
        modelIR: entry.model, modelOrigin: isDemo ? 'demo' : 'catalog', builtInId: id, projectKey: null, projectName: null,
        codeFiles: isDemo ? { ...demoCodeFiles } : {},
        selectedNodeId: isDemo ? 'downsample' : null,
        hoveredNodeId: null, depth: null, toggles: {},
        activeFile: isDemo ? 'blocks.py' : 'model-ir.json', activeLine: null,
        sourceRevision: state.sourceRevision + 1,
        parseError: null, diagnostic: null, logs: [], changed: { ids: [], stamp: Date.now() },
      }))
    },
    loadCodeFiles: (id, files, builtInId) => {
      // A shared link never overwrites your own edits of the same built-in model.
      const pristine = builtInId ? codeModelById[builtInId]?.files : undefined
      if (builtInId && pristine && sameFiles(files, pristine)) { get().loadBuiltIn(builtInId); return }
      loadCode(id, files, builtInId, { fresh: builtInId ? `${builtInId}-shared` : id, name: `${builtInId ? metaFor(builtInId)?.localized?.zh?.name ?? id : id} (${tr(get().locale, 'shared link', '分享链接')})` })
    },
    loadBlank: () => loadCode('my-model', { 'model.py': blankTemplate }, null, { fresh: 'my-model', name: tr(get().locale, 'Blank model', '空白模型') }),
    resetDemo: () => { get().loadBuiltIn('vae3d') },
    updateModelIR: (modelIR) => set({ modelIR }),
    setDepth: (depth) => set({ depth, toggles: {} }),
    toggleCollapsed: (id, collapsed) => {
      const state = get()
      const current = collapsedPredicate(state)(id)
      set({ toggles: { ...state.toggles, [id]: collapsed ?? !current } })
    },
    setLibraryOpen: (libraryOpen) => set({ libraryOpen }),
    openProject: (key) => {
      if (get().projectKey === key) return
      const saved = readProject(key)
      if (!saved) { set({ projects: listProjects() }); return }
      const builtInId = saved.builtInId && codeModelById[saved.builtInId] ? saved.builtInId : null
      loadCode(builtInId ?? key.replace(/^[^:]*:/, ''), saved.codeFiles, builtInId, key)
    },
    deleteProject: (key) => {
      removeProject(key)
      // The open project stays open; it is saved again as soon as it changes.
      set({ projects: listProjects() })
    },
  }
})

/** Writes the current code to "My projects" and the current-view slot. Returns false when the browser refused. */
function saveNow(): boolean {
  clearTimeout(saveTimer)
  if (typeof window === 'undefined') return true
  const current = useModelStore.getState()
  let saved = true
  if (current.modelOrigin === 'code' && current.projectKey) {
    const pristine = current.builtInId ? codeModelById[current.builtInId]?.files : undefined
    // An unchanged built-in model is not a project of yours; only edits are kept.
    if (current.projectKey.startsWith('builtin:') && pristine && sameFiles(current.codeFiles, pristine)) removeProject(current.projectKey)
    else {
      const entry = current.builtInId ? codeModelById[current.builtInId] : undefined
      const name = entry && current.projectKey.startsWith('builtin:') ? (current.locale === 'zh' ? entry.zhName : entry.name) : current.projectName ?? current.modelIR.localized?.[current.locale]?.name ?? current.modelIR.name
      saved = writeProject(current.projectKey, name, current.builtInId, current.codeFiles)
    }
  }
  try {
    // Code models are rebuilt from source on load, so only a light model stub is stored.
    const model = current.modelOrigin === 'code' ? { ...current.modelIR, nodes: current.modelIR.nodes.slice(0, 1).map(({ id, name, type }) => ({ id, name, type })), edges: [] } : current.modelIR
    window.localStorage.setItem(storageKey, JSON.stringify({ origin: current.modelOrigin, builtInId: current.builtInId, locale: current.locale, model, codeFiles: current.codeFiles, projectKey: current.projectKey, projectName: current.projectName }))
  } catch { saved = false }
  const projects = listProjects()
  if (current.saveFailed === saved || JSON.stringify(projects) !== JSON.stringify(current.projects)) useModelStore.setState({ saveFailed: !saved, projects })
  return saved
}

useModelStore.subscribe((state, previous) => {
  if (typeof window === 'undefined') return
  if (state.modelIR === previous.modelIR && state.codeFiles === previous.codeFiles && state.locale === previous.locale && state.modelOrigin === previous.modelOrigin) return
  clearTimeout(saveTimer)
  saveTimer = setTimeout(saveNow, 400)
})

if (typeof window !== 'undefined') window.addEventListener('pagehide', () => { saveNow() })

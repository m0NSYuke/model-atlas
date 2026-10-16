import { create } from 'zustand'
import { useMemo } from 'react'
import type { ModelIR } from '../core/model-ir/types'
import { buildModelFromCode } from '../core/pytorch'
import { codeModelById } from '../data/models/codeCatalog'
import { diffModels, type ModelDiff } from '../core/study/diff'
import { useModelStore } from './modelStore'

// Study tools: shape lens, change comparison, path tracing, notes and the reading checklist.
// Notes and answers are the reader's own and stay in this browser (and in share links / exports).

const storageKey = 'atlas-study-v1'

export type ReadingPass = 1 | 2 | 3

interface Saved {
  shapeLens: boolean
  compare: boolean
  /** modelId → nodeId → note. */
  notes: Record<string, Record<string, string>>
  /** modelId → question key → answer (Ng's four questions, predictions). */
  answers: Record<string, Record<string, string>>
  /** modelId → ticked checklist items. */
  checks: Record<string, string[]>
}

interface StudyState extends Saved {
  /** Second block of a traced path (the first is the selection). */
  pathTarget: string | null
  setShapeLens: (on: boolean) => void
  setCompare: (on: boolean) => void
  setPathTarget: (id: string | null) => void
  setNote: (modelId: string, nodeId: string, text: string) => void
  mergeNotes: (modelId: string, notes: Record<string, string>) => void
  setAnswer: (modelId: string, key: string, text: string) => void
  toggleCheck: (modelId: string, key: string) => void
}

const defaults: Saved = { shapeLens: true, compare: true, notes: {}, answers: {}, checks: {} }

function load(): Saved {
  try { return { ...defaults, ...(JSON.parse(window.localStorage.getItem(storageKey) ?? 'null') ?? {}) } } catch { return defaults }
}

export const useStudyStore = create<StudyState>((set) => ({
  ...load(),
  pathTarget: null,
  setShapeLens: (shapeLens) => set({ shapeLens }),
  setCompare: (compare) => set({ compare }),
  setPathTarget: (pathTarget) => set({ pathTarget }),
  setNote: (modelId, nodeId, text) => set((state) => {
    const notes = { ...(state.notes[modelId] ?? {}) }
    if (text.trim()) notes[nodeId] = text
    else delete notes[nodeId]
    return { notes: { ...state.notes, [modelId]: notes } }
  }),
  mergeNotes: (modelId, incoming) => set((state) => ({ notes: { ...state.notes, [modelId]: { ...(state.notes[modelId] ?? {}), ...incoming } } })),
  setAnswer: (modelId, key, text) => set((state) => ({ answers: { ...state.answers, [modelId]: { ...(state.answers[modelId] ?? {}), [key]: text } } })),
  toggleCheck: (modelId, key) => set((state) => {
    const list = state.checks[modelId] ?? []
    return { checks: { ...state.checks, [modelId]: list.includes(key) ? list.filter((item) => item !== key) : [...list, key] } }
  }),
}))

let saveTimer: ReturnType<typeof setTimeout> | undefined
useStudyStore.subscribe((state) => {
  clearTimeout(saveTimer)
  saveTimer = setTimeout(() => {
    try {
      const { shapeLens, compare, notes, answers, checks } = state
      window.localStorage.setItem(storageKey, JSON.stringify({ shapeLens, compare, notes, answers, checks }))
    } catch { /* Storage may be unavailable; the session still works. */ }
  }, 300)
})

// A new selection ends a traced path unless it is the path's own start.
useModelStore.subscribe((state, previous) => {
  if (state.selectedNodeId !== previous.selectedNodeId || state.modelIR.id !== previous.modelIR.id) useStudyStore.getState().setPathTarget(null)
})

/** Notes are keyed by the built-in model when there is one, so they survive code edits. */
export const notesKey = (state: { builtInId: string | null; modelIR: ModelIR }) => state.builtInId ?? state.modelIR.id

const baselines = new Map<string, ModelIR | null>()
/** The architecture traced from the original, unedited code of a built-in model. */
export function baselineFor(builtInId: string | null): ModelIR | null {
  if (!builtInId || !codeModelById[builtInId]) return null
  if (!baselines.has(builtInId)) {
    const entry = codeModelById[builtInId]
    try { baselines.set(builtInId, buildModelFromCode(entry.id, entry.files).model) } catch { baselines.set(builtInId, null) }
  }
  return baselines.get(builtInId) ?? null
}

/** What the current edits changed in the architecture, or null when the code is the original. */
export function useModelDiff(): ModelDiff | null {
  const model = useModelStore((state) => state.modelIR)
  const builtInId = useModelStore((state) => state.builtInId)
  const codeFiles = useModelStore((state) => state.codeFiles)
  const origin = useModelStore((state) => state.modelOrigin)
  return useMemo(() => {
    if (origin !== 'code' || !builtInId || !codeModelById[builtInId]) return null
    const entry = codeModelById[builtInId]
    if (Object.entries(entry.files).every(([file, source]) => codeFiles[file] === source)) return null
    const base = baselineFor(builtInId)
    return base ? diffModels(base, model) : null
  }, [model, builtInId, codeFiles, origin])
}

/** Diff marks to draw on blocks, when comparison is on. */
export function useDiffMarks() {
  const diff = useModelDiff()
  const compare = useStudyStore((state) => state.compare)
  return compare && diff ? diff.marks : null
}

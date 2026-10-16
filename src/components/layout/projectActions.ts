import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import { notesKey, useStudyStore } from '../../store/studyStore'
import { codeModelById } from '../../data/models/codeCatalog'
import { studyById } from '../../data/models/studyCatalog'
import { decodeShare, encodeShare, shareParam, type SharedState } from '../../core/study/share'

/** Download the current model and its code as an Atlas project JSON, with the reader's notes. */
export function exportProject() {
  const state = useModelStore.getState()
  const { modelIR: model, codeFiles } = state
  const study = useStudyStore.getState()
  const key = notesKey(state)
  const contents = JSON.stringify({ format: 'model-atlas', version: 2, model, codeFiles, study: { notes: study.notes[key] ?? {}, answers: study.answers[key] ?? {} } }, null, 2)
  const url = URL.createObjectURL(new Blob([contents], { type: 'application/json' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `${model.id.replace(/[^\w-]/g, '-') || 'model'}-atlas.json`
  link.click()
  URL.revokeObjectURL(url)
}

/** A link that reopens exactly this view: model, code edits, selection, traced path, depth and notes. */
export async function createShareLink(): Promise<string | null> {
  const state = useModelStore.getState()
  if (state.modelOrigin === 'import') return null
  const entry = state.builtInId ? codeModelById[state.builtInId] : undefined
  const files = entry ? Object.fromEntries(Object.entries(state.codeFiles).filter(([file, source]) => entry.files[file] !== source))
    : state.modelOrigin === 'code' ? state.codeFiles : {}
  const shared: SharedState = {
    v: 1,
    model: state.builtInId,
    files: Object.keys(files).length ? files : undefined,
    selected: state.selectedNodeId,
    view: useUiStore.getState().view,
    depth: state.depth,
    notes: useStudyStore.getState().notes[notesKey(state)],
    path: useStudyStore.getState().pathTarget,
  }
  const token = await encodeShare(shared)
  const url = `${window.location.origin}${window.location.pathname}#${shareParam}=${token}`
  try { await navigator.clipboard.writeText(url) } catch { /* Clipboard may be blocked; the link is still returned. */ }
  return url
}

/** Opens a shared link found in the address bar, then removes it so a reload keeps the user's own edits. */
export async function openSharedLink(): Promise<boolean> {
  const match = window.location.hash.match(new RegExp(`^#${shareParam}=(.+)$`))
  if (!match) return false
  // Take the link out of the address bar first, so a second call (React strict mode) finds nothing.
  window.history.replaceState(null, '', window.location.pathname + window.location.search)
  const shared = await decodeShare(match[1])
  if (!shared) return false
  const store = useModelStore.getState()
  if (shared.model && codeModelById[shared.model]) store.loadCodeFiles(shared.model, { ...codeModelById[shared.model].files, ...(shared.files ?? {}) }, shared.model)
  else if (shared.model && studyById[shared.model]) store.loadBuiltIn(shared.model)
  else if (shared.files && Object.keys(shared.files).length) store.loadCodeFiles('shared-model', shared.files, null)
  else return false
  if (shared.depth !== undefined) useModelStore.getState().setDepth(shared.depth)
  if (shared.notes) useStudyStore.getState().mergeNotes(notesKey(useModelStore.getState()), shared.notes)
  if (shared.view) useUiStore.getState().setView(shared.view)
  if (shared.selected) useModelStore.getState().selectNode(shared.selected)
  if (shared.path) useStudyStore.getState().setPathTarget(shared.path)
  return true
}

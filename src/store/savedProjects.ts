// "My projects": every imported project, blank model and edited built-in model is kept in this browser,
// one localStorage entry per project, so switching models never loses code.

export interface ProjectInfo {
  key: string
  name: string
  /** The built-in model this project started from, when any. */
  builtInId: string | null
  updatedAt: number
  fileCount: number
  bytes: number
}

export interface SavedProjectFiles extends ProjectInfo {
  codeFiles: Record<string, string>
}

const indexKey = 'model-atlas-projects-v1'
const entryKey = (key: string) => `${indexKey}:${key}`

function storage(): Storage | null {
  try { return typeof window === 'undefined' ? null : window.localStorage } catch { return null }
}

export function listProjects(): ProjectInfo[] {
  try {
    const list = JSON.parse(storage()?.getItem(indexKey) ?? '[]') as ProjectInfo[]
    return Array.isArray(list) ? list.filter((item) => item && typeof item.key === 'string').sort((a, b) => b.updatedAt - a.updatedAt) : []
  } catch { return [] }
}

export function readProject(key: string): SavedProjectFiles | null {
  const info = listProjects().find((item) => item.key === key)
  if (!info) return null
  try {
    const codeFiles = JSON.parse(storage()?.getItem(entryKey(key)) ?? 'null') as Record<string, string> | null
    return codeFiles && typeof codeFiles === 'object' ? { ...info, codeFiles } : null
  } catch { return null }
}

/** Saves one project. Returns false when the browser refuses (storage full or disabled). */
export function writeProject(key: string, name: string, builtInId: string | null, codeFiles: Record<string, string>): boolean {
  const store = storage()
  if (!store) return false
  const body = JSON.stringify(codeFiles)
  const info: ProjectInfo = { key, name, builtInId, updatedAt: Date.now(), fileCount: Object.keys(codeFiles).length, bytes: body.length }
  try {
    store.setItem(entryKey(key), body)
    store.setItem(indexKey, JSON.stringify([info, ...listProjects().filter((item) => item.key !== key)]))
    return true
  } catch { return false }
}

export function removeProject(key: string): void {
  const store = storage()
  if (!store) return
  try {
    store.removeItem(entryKey(key))
    store.setItem(indexKey, JSON.stringify(listProjects().filter((item) => item.key !== key)))
  } catch { /* Nothing more can be done when storage is unavailable. */ }
}

/** A key not used by any saved project: `name`, `name-2`, `name-3` … */
export function uniqueProjectKey(base: string, taken: string[] = []): string {
  const used = new Set([...listProjects().map((item) => item.key), ...taken])
  const stem = `project:${base.toLowerCase().replace(/[^\w-]+/g, '-').replace(/^-+|-+$/g, '') || 'my-model'}`
  if (!used.has(stem)) return stem
  let index = 2
  while (used.has(`${stem}-${index}`)) index++
  return `${stem}-${index}`
}

export const builtInProjectKey = (id: string) => `builtin:${id}`

export function sameFiles(a: Record<string, string>, b: Record<string, string>): boolean {
  const keys = Object.keys(a)
  return keys.length === Object.keys(b).length && keys.every((key) => a[key] === b[key])
}

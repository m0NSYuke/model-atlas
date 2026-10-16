// Share links: the whole working state (code edits, selection, view, notes) packed into the URL hash,
// compressed with the browser's own deflate, so a link reproduces exactly what the sender was looking at.

// The .ts extension lets the study tests load this file directly with Node's type stripping.
import { codeFilesOrNull, isPlainObject, stringRecord } from '../validate.ts'

export interface SharedState {
  v: 1
  /** Built-in model id, or null for the user's own code. */
  model: string | null
  /** Code files when they differ from the built-in model (all files for the user's own code). */
  files?: Record<string, string>
  selected?: string | null
  view?: '3d' | '2d'
  depth?: number | null
  notes?: Record<string, string>
  path?: string | null
}

const toBase64Url = (bytes: Uint8Array) => {
  let binary = ''
  bytes.forEach((byte) => { binary += String.fromCharCode(byte) })
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
const fromBase64Url = (text: string) => {
  const binary = atob(text.replace(/-/g, '+').replace(/_/g, '/'))
  return Uint8Array.from(binary, (char) => char.charCodeAt(0))
}

async function pipe(bytes: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
  const output = new Blob([bytes as Uint8Array<ArrayBuffer>]).stream().pipeThrough(stream)
  return new Uint8Array(await new Response(output).arrayBuffer())
}

export async function encodeShare(state: SharedState): Promise<string> {
  const json = new TextEncoder().encode(JSON.stringify(state))
  if (typeof CompressionStream === 'undefined') return `j${toBase64Url(json)}`
  return `z${toBase64Url(await pipe(json, new CompressionStream('deflate-raw')))}`
}

export async function decodeShare(token: string): Promise<SharedState | null> {
  try {
    const bytes = fromBase64Url(token.slice(1))
    const json = token[0] === 'z' ? await pipe(bytes, new DecompressionStream('deflate-raw')) : bytes
    return sanitize(JSON.parse(new TextDecoder().decode(json)))
  } catch { return null }
}

/** Keeps only well-typed fields: a link is untrusted input, and what it carries is saved in this browser. */
function sanitize(raw: unknown): SharedState | null {
  if (!isPlainObject(raw) || raw.v !== 1) return null
  const text = (value: unknown) => typeof value === 'string' ? value : null
  const state: SharedState = { v: 1, model: text(raw.model) }
  const files = codeFilesOrNull(raw.files)
  if (files) state.files = files
  const notes = stringRecord(raw.notes)
  if (notes) state.notes = notes
  if (raw.selected !== undefined) state.selected = text(raw.selected)
  if (raw.path !== undefined) state.path = text(raw.path)
  if (raw.view === '3d' || raw.view === '2d') state.view = raw.view
  if (raw.depth === null || (typeof raw.depth === 'number' && Number.isInteger(raw.depth) && raw.depth >= 0)) state.depth = raw.depth
  return state
}

export const shareParam = 's'

// Share links: the whole working state (code edits, selection, view, notes) packed into the URL hash,
// compressed with the browser's own deflate, so a link reproduces exactly what the sender was looking at.

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
    const state = JSON.parse(new TextDecoder().decode(json)) as SharedState
    return state && state.v === 1 ? state : null
  } catch { return null }
}

export const shareParam = 's'

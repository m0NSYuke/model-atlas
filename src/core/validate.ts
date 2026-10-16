// Checks for data that comes from outside the app (localStorage, share links, imported files),
// so a value of the wrong shape falls back to a default instead of crashing a render.

export const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

/** The string-valued entries of an object, or null when the value is not an object. */
export function stringRecord(value: unknown): Record<string, string> | null {
  if (!isPlainObject(value)) return null
  return Object.fromEntries(Object.entries(value).filter(([, text]) => typeof text === 'string')) as Record<string, string>
}

/** Code files: a non-empty object of file name → source text, or null. */
export function codeFilesOrNull(value: unknown): Record<string, string> | null {
  const files = stringRecord(value)
  return files && Object.keys(files).length ? files : null
}

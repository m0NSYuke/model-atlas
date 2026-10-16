// Which lines of a file differ from the original: a plain LCS line diff, enough for a few hundred lines.

export interface LineChanges {
  /** Lines (1-based, in the edited text) that are new. */
  added: number[]
  /** Lines that replace original lines. */
  modified: number[]
  /** Lines after which original lines were deleted (0 = before the first line). */
  deleted: number[]
}

export function lineChanges(original: string, edited: string): LineChanges {
  const a = original.split('\n')
  const b = edited.split('\n')
  const empty = { added: [], modified: [], deleted: [] }
  if (original === edited) return empty
  // Trim the common head and tail; the table is only built for the changed middle.
  let head = 0
  while (head < a.length && head < b.length && a[head] === b[head]) head++
  let tail = 0
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++
  const x = a.slice(head, a.length - tail)
  const y = b.slice(head, b.length - tail)
  if (x.length * y.length > 400_000) return { ...empty, modified: y.map((_, index) => head + index + 1) }
  const table = Array.from({ length: x.length + 1 }, () => new Uint16Array(y.length + 1))
  for (let i = x.length - 1; i >= 0; i--) for (let j = y.length - 1; j >= 0; j--) table[i][j] = x[i] === y[j] ? table[i + 1][j + 1] + 1 : Math.max(table[i + 1][j], table[i][j + 1])
  // Edit script: '=' keep, '-' remove an original line, '+' insert an edited line.
  const ops: ('=' | '-' | '+')[] = []
  let i = 0
  let j = 0
  while (i < x.length && j < y.length) {
    if (x[i] === y[j]) { ops.push('='); i++; j++ } else if (table[i + 1][j] >= table[i][j + 1]) { ops.push('-'); i++ } else { ops.push('+'); j++ }
  }
  while (i++ < x.length) ops.push('-')
  while (j++ < y.length) ops.push('+')
  // A hunk of removals and insertions reads as modified lines; leftovers are additions or a deletion mark.
  const added: number[] = []
  const modified: number[] = []
  const deleted: number[] = []
  let line = head
  for (let index = 0; index < ops.length;) {
    if (ops[index] === '=') { line++; index++; continue }
    let removedRun = 0
    const inserted: number[] = []
    while (index < ops.length && ops[index] !== '=') { if (ops[index] === '-') removedRun++; else inserted.push(++line); index++ }
    const paired = Math.min(removedRun, inserted.length)
    modified.push(...inserted.slice(0, paired))
    added.push(...inserted.slice(paired))
    if (removedRun > paired) deleted.push(line)
  }
  return { added, modified, deleted }
}

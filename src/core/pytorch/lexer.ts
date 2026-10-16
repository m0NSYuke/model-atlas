// Tokenizer for the subset of Python used by PyTorch model definitions.
// Produces NEWLINE / INDENT / DEDENT tokens like CPython so the parser can stay simple.

export type TokenKind = 'name' | 'number' | 'string' | 'op' | 'newline' | 'indent' | 'dedent' | 'eof'

export interface Token {
  kind: TokenKind
  value: string
  line: number
  col: number
  /** f-string: `value` keeps the raw template (escapes unprocessed) so the parser can split out `{expr}` fields. */
  fstring?: boolean
  raw?: boolean
}

export class PySyntaxError extends Error {
  constructor(message: string, public line: number, public file?: string) {
    super(message)
  }
}

const operators = [
  '**=', '//=', '>>=', '<<=', '...', '->', ':=',
  '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '@=', '&=', '|=', '^=', '<<', '>>',
  '+', '-', '*', '/', '%', '@', '<', '>', '=', '(', ')', '[', ']', '{', '}', ',', ':', '.', ';', '|', '&', '^', '~',
]

export function tokenize(source: string, file = '<code>'): Token[] {
  const tokens: Token[] = []
  const indents = [0]
  let depth = 0
  let index = 0
  let line = 1
  let lineStart = 0
  let atLineStart = true
  const text = source.replace(/\r\n?/g, '\n')

  const push = (kind: TokenKind, value: string, startIndex: number) => tokens.push({ kind, value, line, col: startIndex - lineStart + 1 })

  while (index < text.length) {
    if (atLineStart && depth === 0) {
      // Measure indentation of a logical line; skip blank and comment-only lines.
      let width = 0
      let cursor = index
      while (cursor < text.length && (text[cursor] === ' ' || text[cursor] === '\t')) { width += text[cursor] === '\t' ? 4 : 1; cursor++ }
      if (cursor >= text.length) { index = cursor; break }
      if (text[cursor] === '\n' || text[cursor] === '#') {
        while (cursor < text.length && text[cursor] !== '\n') cursor++
        index = cursor + 1
        line++
        lineStart = index
        continue
      }
      index = cursor
      atLineStart = false
      if (width > indents[indents.length - 1]) { indents.push(width); push('indent', '', index) }
      else {
        while (width < indents[indents.length - 1]) { indents.pop(); push('dedent', '', index) }
        if (width !== indents[indents.length - 1]) throw new PySyntaxError('Inconsistent indentation', line, file)
      }
    }
    const char = text[index]
    if (char === '\n') {
      if (depth === 0 && tokens.length && tokens[tokens.length - 1].kind !== 'newline') push('newline', '', index)
      index++
      line++
      lineStart = index
      atLineStart = depth === 0
      continue
    }
    if (char === ' ' || char === '\t') { index++; continue }
    if (char === '#') { while (index < text.length && text[index] !== '\n') index++; continue }
    if (char === '\\' && text[index + 1] === '\n') { index += 2; line++; lineStart = index; continue }

    // Strings, including prefixes (r, b, f, rb …) and triple quotes.
    const prefix = /^([rRbBuUfF]{0,2})('''|"""|'|")/.exec(text.slice(index, index + 5))
    if (prefix) {
      const start = index
      const startLine = line
      const quote = prefix[2]
      const raw = /[rR]/.test(prefix[1])
      const fstring = /[fF]/.test(prefix[1])
      index += prefix[0].length
      let value = ''
      while (true) {
        if (index >= text.length) throw new PySyntaxError('Unterminated string', startLine, file)
        if (text.startsWith(quote, index)) { index += quote.length; break }
        const current = text[index]
        if (current === '\n') {
          if (quote.length === 1) throw new PySyntaxError('Unterminated string', startLine, file)
          line++
          lineStart = index + 1
        }
        if (current === '\\' && fstring && index + 1 < text.length) {
          // Keep the escape for the parser; it unescapes only the literal parts of an f-string.
          if (text[index + 1] === '\n') { line++; lineStart = index + 2 }
          value += current + text[index + 1]
          index += 2
          continue
        }
        if (current === '\\' && !raw && index + 1 < text.length) {
          const next = text[index + 1]
          value += next === 'n' ? '\n' : next === 't' ? '\t' : next === '\n' ? '' : next
          if (next === '\n') { line++; lineStart = index + 2 }
          index += 2
          continue
        }
        value += current
        index++
      }
      tokens.push({ kind: 'string', value, line: startLine, col: start - lineStart + 1, ...(fstring ? { fstring, raw } : {}) })
      continue
    }
    if (/[A-Za-z_À-￿]/.test(char)) {
      const start = index
      while (index < text.length && /[A-Za-z0-9_À-￿]/.test(text[index])) index++
      push('name', text.slice(start, index), start)
      continue
    }
    if (/[0-9]/.test(char) || (char === '.' && /[0-9]/.test(text[index + 1] ?? ''))) {
      const match = /^(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d+)?j?)/.exec(text.slice(index))!
      push('number', match[0].replace(/_/g, ''), index)
      index += match[0].length
      continue
    }
    const op = operators.find((candidate) => text.startsWith(candidate, index))
    if (op) {
      if (op === '(' || op === '[' || op === '{') depth++
      if (op === ')' || op === ']' || op === '}') depth = Math.max(0, depth - 1)
      push('op', op, index)
      index += op.length
      continue
    }
    throw new PySyntaxError(`Unexpected character '${char}'`, line, file)
  }
  if (tokens.length && tokens[tokens.length - 1].kind !== 'newline') push('newline', '', index)
  while (indents.length > 1) { indents.pop(); push('dedent', '', index) }
  push('eof', '', index)
  return tokens
}

/** Processes backslash escapes the way the tokenizer does for ordinary strings. */
export function unescapePython(text: string): string {
  let value = ''
  for (let index = 0; index < text.length; index++) {
    const current = text[index]
    if (current === '\\' && index + 1 < text.length) {
      const next = text[++index]
      value += next === 'n' ? '\n' : next === 't' ? '\t' : next === '\n' ? '' : next
    } else value += current
  }
  return value
}

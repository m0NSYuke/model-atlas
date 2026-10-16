// Recursive-descent parser for a practical subset of Python 3.
import { PySyntaxError, tokenize, unescapePython, type Token } from './lexer'

export type Expr =
  | { kind: 'name'; id: string; line: number }
  | { kind: 'const'; value: number | string | boolean | null; line: number }
  | { kind: 'ellipsis'; line: number }
  | { kind: 'attr'; object: Expr; name: string; line: number }
  | { kind: 'subscript'; object: Expr; index: Expr; line: number }
  | { kind: 'slice'; lower?: Expr; upper?: Expr; step?: Expr; line: number }
  | { kind: 'call'; func: Expr; args: Arg[]; line: number; endLine: number }
  | { kind: 'binop'; op: string; left: Expr; right: Expr; line: number }
  | { kind: 'unary'; op: string; operand: Expr; line: number }
  | { kind: 'bool'; op: 'and' | 'or'; values: Expr[]; line: number }
  | { kind: 'compare'; left: Expr; ops: string[]; comparators: Expr[]; line: number }
  | { kind: 'ifexp'; test: Expr; body: Expr; orelse: Expr; line: number }
  | { kind: 'tuple'; items: Expr[]; line: number }
  | { kind: 'list'; items: Expr[]; line: number }
  | { kind: 'dict'; keys: (Expr | null)[]; values: Expr[]; line: number }
  | { kind: 'set'; items: Expr[]; line: number }
  | { kind: 'comp'; type: 'list' | 'gen' | 'set' | 'dict'; element: Expr; value?: Expr; target: Expr; iter: Expr; conditions: Expr[]; line: number; nested?: boolean }
  | { kind: 'lambda'; params: Param[]; body: Expr; line: number }
  | { kind: 'starred'; value: Expr; line: number }
  | { kind: 'fstring'; parts: FStringPart[]; line: number }
  | { kind: 'namedexpr'; target: string; value: Expr; line: number }

export interface Arg { name?: string; value: Expr; star?: '*' | '**' }
export interface Param { name: string; defaultValue?: Expr; star?: '*' | '**'; annotation?: Expr }

export type Stmt =
  | { kind: 'expr'; value: Expr; line: number; endLine: number }
  | { kind: 'assign'; targets: Expr[]; value: Expr; line: number; endLine: number }
  | { kind: 'augassign'; target: Expr; op: string; value: Expr; line: number; endLine: number }
  | { kind: 'annassign'; target: Expr; annotation: Expr; value?: Expr; line: number; endLine: number }
  | { kind: 'return'; value?: Expr; line: number; endLine: number }
  | { kind: 'if'; test: Expr; body: Stmt[]; orelse: Stmt[]; line: number; endLine: number }
  | { kind: 'for'; target: Expr; iter: Expr; body: Stmt[]; orelse: Stmt[]; line: number; endLine: number }
  | { kind: 'while'; test: Expr; body: Stmt[]; orelse: Stmt[]; line: number; endLine: number }
  | { kind: 'with'; items: { value: Expr; target?: Expr }[]; body: Stmt[]; line: number; endLine: number }
  | { kind: 'try'; body: Stmt[]; handlers: Stmt[][]; orelse: Stmt[]; finalbody: Stmt[]; line: number; endLine: number }
  | { kind: 'def'; name: string; params: Param[]; body: Stmt[]; decorators: Expr[]; docstring?: string; line: number; endLine: number }
  | { kind: 'class'; name: string; bases: Expr[]; body: Stmt[]; decorators: Expr[]; docstring?: string; line: number; endLine: number }
  | { kind: 'raise'; value?: Expr; line: number; endLine: number }
  | { kind: 'assert'; test: Expr; message?: Expr; line: number; endLine: number }
  | { kind: 'import'; names: ImportName[]; line: number; endLine: number }
  | { kind: 'global' | 'nonlocal'; names: string[]; line: number; endLine: number }
  | { kind: 'del'; targets: Expr[]; line: number; endLine: number }
  | { kind: 'pass' | 'break' | 'continue'; line: number; endLine: number }

/** A literal piece or a `{expr!conversion:spec}` field of an f-string; `debug` holds the `expr=` text. */
export type FStringPart = string | { expr: Expr; conversion?: 'r' | 's' | 'a'; spec?: FStringPart[]; debug?: string }

export interface ImportName { module: string; name?: string; asName: string }

const augOps = new Set(['+=', '-=', '*=', '/=', '//=', '%=', '**=', '@=', '&=', '|=', '^=', '>>=', '<<='])
const compareOps = new Set(['<', '>', '==', '!=', '<=', '>='])

export class Parser {
  private position = 0
  private tokens: Token[]

  constructor(source: string, private file: string) {
    this.tokens = tokenize(source, file)
  }

  private get token(): Token { return this.tokens[this.position] }
  private kindAt(): Token['kind'] { return this.tokens[this.position].kind }
  private peek(offset = 1): Token { return this.tokens[Math.min(this.position + offset, this.tokens.length - 1)] }
  private previousLine(): number { return this.tokens[Math.max(0, this.position - 1)].line }
  private fail(message: string, token = this.token): never { throw new PySyntaxError(message, token.line, this.file) }
  private isOp(value: string): boolean { return this.token.kind === 'op' && this.token.value === value }
  private isName(value: string): boolean { return this.token.kind === 'name' && this.token.value === value }
  private accept(value: string): boolean {
    if ((this.token.kind === 'op' || this.token.kind === 'name') && this.token.value === value) { this.position++; return true }
    return false
  }
  private expect(value: string): Token {
    const token = this.token
    if (!this.accept(value)) this.fail(`Expected '${value}' but found '${token.value || token.kind}'`)
    return token
  }
  private expectName(): string {
    const token = this.token
    if (token.kind !== 'name') this.fail(`Expected a name but found '${token.value || token.kind}'`)
    this.position++
    return token.value
  }

  parseModule(): Stmt[] {
    const body: Stmt[] = []
    while (this.token.kind !== 'eof') {
      if (this.token.kind === 'newline') { this.position++; continue }
      body.push(...this.statement())
    }
    return body
  }

  private block(): Stmt[] {
    this.expect(':')
    if (this.token.kind !== 'newline') {
      const statements = this.simpleStatements()
      return statements
    }
    this.position++
    if (this.kindAt() !== 'indent') this.fail('Expected an indented block')
    this.position++
    const body: Stmt[] = []
    while (this.kindAt() !== 'dedent' && this.kindAt() !== 'eof') {
      if (this.kindAt() === 'newline') { this.position++; continue }
      body.push(...this.statement())
    }
    if (this.kindAt() === 'dedent') this.position++
    return body
  }

  private docstring(body: Stmt[]): string | undefined {
    const first = body[0]
    return first?.kind === 'expr' && first.value.kind === 'const' && typeof first.value.value === 'string' ? first.value.value : undefined
  }

  private statement(): Stmt[] {
    const token = this.token
    if (this.isOp('@')) {
      const decorators: Expr[] = []
      while (this.accept('@')) {
        decorators.push(this.expression())
        if (this.token.kind === 'newline') this.position++
      }
      return [this.withDecorators(decorators)]
    }
    if (token.kind === 'name') {
      switch (token.value) {
        case 'def': case 'class': return [this.withDecorators([])]
        case 'if': return [this.ifStatement()]
        case 'for': {
          this.position++
          const target = this.targetList()
          this.expect('in')
          const iter = this.testList()
          const body = this.block()
          let orelse: Stmt[] = []
          if (this.isName('else') && this.peek().value === ':') { this.position++; orelse = this.block() }
          return [{ kind: 'for', target, iter, body, orelse, line: token.line, endLine: this.previousLine() }]
        }
        case 'while': {
          this.position++
          const test = this.test()
          const body = this.block()
          let orelse: Stmt[] = []
          if (this.isName('else')) { this.position++; orelse = this.block() }
          return [{ kind: 'while', test, body, orelse, line: token.line, endLine: this.previousLine() }]
        }
        case 'with': {
          this.position++
          const items: { value: Expr; target?: Expr }[] = []
          do {
            const value = this.test()
            const target = this.accept('as') ? this.target() : undefined
            items.push({ value, target })
          } while (this.accept(','))
          const body = this.block()
          return [{ kind: 'with', items, body, line: token.line, endLine: this.previousLine() }]
        }
        case 'try': {
          this.position++
          const body = this.block()
          const handlers: Stmt[][] = []
          let orelse: Stmt[] = []
          let finalbody: Stmt[] = []
          while (this.isName('except')) {
            this.position++
            if (!this.isOp(':')) { this.test(); if (this.accept('as')) this.expectName() }
            handlers.push(this.block())
          }
          if (this.isName('else')) { this.position++; orelse = this.block() }
          if (this.isName('finally')) { this.position++; finalbody = this.block() }
          return [{ kind: 'try', body, handlers, orelse, finalbody, line: token.line, endLine: this.previousLine() }]
        }
      }
    }
    return this.simpleStatements()
  }

  private withDecorators(decorators: Expr[]): Stmt {
    const token = this.token
    if (this.accept('def')) {
      const name = this.expectName()
      this.expect('(')
      const params = this.parameters(')')
      this.expect(')')
      if (this.accept('->')) this.test()
      const body = this.block()
      return { kind: 'def', name, params, body, decorators, docstring: this.docstring(body), line: token.line, endLine: this.previousLine() }
    }
    if (this.accept('class')) {
      const name = this.expectName()
      let bases: Expr[] = []
      if (this.accept('(')) {
        bases = this.callArguments().filter((arg) => !arg.name).map((arg) => arg.value)
        this.expect(')')
      }
      const body = this.block()
      return { kind: 'class', name, bases, body, decorators, docstring: this.docstring(body), line: token.line, endLine: this.previousLine() }
    }
    return this.fail('Expected def or class after decorator')
  }

  private ifStatement(): Stmt {
    const token = this.token
    this.position++
    const test = this.test()
    const body = this.block()
    let orelse: Stmt[] = []
    if (this.isName('elif')) orelse = [this.ifStatement()]
    else if (this.isName('else')) { this.position++; orelse = this.block() }
    return { kind: 'if', test, body, orelse, line: token.line, endLine: this.previousLine() }
  }

  private parameters(closing: string): Param[] {
    const params: Param[] = []
    while (!this.isOp(closing)) {
      if (this.accept('/')) { if (!this.accept(',')) break; continue }
      let star: Param['star']
      if (this.accept('**')) star = '**'
      else if (this.accept('*')) {
        star = '*'
        if (this.isOp(',') || this.isOp(closing)) { this.accept(','); continue }
      }
      const name = this.expectName()
      const annotation = closing === ')' && this.accept(':') ? this.test() : undefined
      const defaultValue = this.accept('=') ? this.test() : undefined
      params.push({ name, defaultValue, star, annotation })
      if (!this.accept(',')) break
    }
    return params
  }

  private simpleStatements(): Stmt[] {
    const statements = [this.smallStatement()]
    while (this.accept(';')) { if (this.token.kind === 'newline') break; statements.push(this.smallStatement()) }
    if (this.token.kind === 'newline') this.position++
    else if (this.token.kind !== 'eof' && this.token.kind !== 'dedent') this.fail(`Unexpected '${this.token.value}'`)
    return statements
  }

  private smallStatement(): Stmt {
    const token = this.token
    const line = token.line
    const end = () => this.previousLine()
    if (token.kind === 'name') {
      switch (token.value) {
        case 'pass': case 'break': case 'continue': this.position++; return { kind: token.value, line, endLine: line }
        case 'import': case 'from': return { kind: 'import', names: this.importNames(), line, endLine: end() }
        case 'global': case 'nonlocal': {
          this.position++
          const names = [this.expectName()]
          while (this.accept(',')) names.push(this.expectName())
          return { kind: token.value, names, line, endLine: end() }
        }
        case 'del': {
          this.position++
          const targets = [this.target()]
          while (this.accept(',') && this.token.kind !== 'newline' && this.token.kind !== 'eof' && !this.isOp(';')) targets.push(this.target())
          return { kind: 'del', targets, line, endLine: end() }
        }
        case 'return': {
          this.position++
          const value = this.token.kind === 'newline' || this.isOp(';') || this.token.kind === 'eof' ? undefined : this.testList(true)
          return { kind: 'return', value, line, endLine: end() }
        }
        case 'raise': {
          this.position++
          const value = this.token.kind === 'newline' || this.token.kind === 'eof' ? undefined : this.test()
          if (this.accept('from')) this.test()
          return { kind: 'raise', value, line, endLine: end() }
        }
        case 'assert': {
          this.position++
          const test = this.test()
          const message = this.accept(',') ? this.test() : undefined
          return { kind: 'assert', test, message, line, endLine: end() }
        }
      }
    }
    const first = this.testList(true)
    if (this.isOp(':') ) {
      this.position++
      const annotation = this.test()
      const value = this.accept('=') ? this.testList(true) : undefined
      return { kind: 'annassign', target: first, annotation, value, line, endLine: end() }
    }
    if (this.token.kind === 'op' && augOps.has(this.token.value)) {
      const op = this.token.value.slice(0, -1)
      this.position++
      const value = this.testList(true)
      return { kind: 'augassign', target: first, op, value, line, endLine: end() }
    }
    if (this.isOp('=')) {
      const targets = [first]
      let value: Expr = first
      while (this.accept('=')) {
        value = this.testList(true)
        targets.push(value)
      }
      targets.pop()
      return { kind: 'assign', targets, value, line, endLine: end() }
    }
    return { kind: 'expr', value: first, line, endLine: end() }
  }

  private dottedName(): string {
    let name = ''
    while (this.isOp('.')) { this.position++; name += '.' }
    if (this.token.kind === 'name') {
      name += this.expectName()
      while (this.accept('.')) name += '.' + this.expectName()
    }
    return name
  }

  private importNames(): ImportName[] {
    const names: ImportName[] = []
    if (this.accept('import')) {
      do {
        const module = this.dottedName()
        if (this.accept('as')) names.push({ module, asName: this.expectName() })
        else names.push({ module: module.split('.')[0], asName: module.split('.')[0] })
      } while (this.accept(','))
      return names
    }
    this.expect('from')
    const module = this.dottedName()
    this.expect('import')
    const parenthesized = this.accept('(')
    do {
      if (this.isOp(')')) break
      if (this.accept('*')) { names.push({ module, name: '*', asName: '*' }); continue }
      const name = this.expectName()
      names.push({ module, name, asName: this.accept('as') ? this.expectName() : name })
    } while (this.accept(','))
    if (parenthesized) this.expect(')')
    return names
  }

  private target(): Expr { return this.orExpression() }

  private targetList(): Expr {
    const line = this.token.line
    const first = this.starOr(() => this.orExpression())
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.accept(',')) {
      if (this.isName('in') || this.isOp('=')) break
      items.push(this.starOr(() => this.orExpression()))
    }
    return { kind: 'tuple', items, line }
  }

  private starOr(parse: () => Expr): Expr {
    const line = this.token.line
    if (this.accept('*')) return { kind: 'starred', value: parse(), line }
    return parse()
  }

  testList(allowStar = false): Expr {
    const line = this.token.line
    const first = allowStar ? this.starOr(() => this.test()) : this.test()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.accept(',')) {
      if (this.token.kind === 'newline' || this.isOp('=') || this.isOp(')') || this.isOp(':') || this.token.kind === 'eof' || this.isOp(';')) break
      items.push(allowStar ? this.starOr(() => this.test()) : this.test())
    }
    return { kind: 'tuple', items, line }
  }

  expression(): Expr { return this.test() }

  private test(): Expr {
    const line = this.token.line
    if (this.token.kind === 'name' && this.peek().kind === 'op' && this.peek().value === ':=') {
      const target = this.expectName()
      this.position++
      return { kind: 'namedexpr', target, value: this.test(), line }
    }
    if (this.isName('lambda')) {
      this.position++
      const params = this.parameters(':')
      this.expect(':')
      return { kind: 'lambda', params, body: this.test(), line }
    }
    const body = this.orTest()
    if (this.isName('if') ) {
      // Ternary only when an else follows on the same expression.
      const save = this.position
      this.position++
      const test = this.orTest()
      if (this.accept('else')) return { kind: 'ifexp', test, body, orelse: this.test(), line }
      this.position = save
    }
    return body
  }

  private orTest(): Expr {
    const line = this.token.line
    const values = [this.andTest()]
    while (this.accept('or')) values.push(this.andTest())
    return values.length === 1 ? values[0] : { kind: 'bool', op: 'or', values, line }
  }

  private andTest(): Expr {
    const line = this.token.line
    const values = [this.notTest()]
    while (this.accept('and')) values.push(this.notTest())
    return values.length === 1 ? values[0] : { kind: 'bool', op: 'and', values, line }
  }

  private notTest(): Expr {
    const line = this.token.line
    if (this.accept('not')) return { kind: 'unary', op: 'not', operand: this.notTest(), line }
    return this.comparison()
  }

  private comparison(): Expr {
    const line = this.token.line
    const left = this.orExpression()
    const ops: string[] = []
    const comparators: Expr[] = []
    while (true) {
      if (this.token.kind === 'op' && compareOps.has(this.token.value)) { ops.push(this.token.value); this.position++ }
      else if (this.isName('in')) { ops.push('in'); this.position++ }
      else if (this.isName('not') && this.peek().value === 'in') { ops.push('not in'); this.position += 2 }
      else if (this.isName('is')) { this.position++; ops.push(this.accept('not') ? 'is not' : 'is') }
      else break
      comparators.push(this.orExpression())
    }
    return ops.length ? { kind: 'compare', left, ops, comparators, line } : left
  }

  private binaryLevel(operators: string[], next: () => Expr): Expr {
    let left = next()
    while (this.token.kind === 'op' && operators.includes(this.token.value)) {
      const op = this.token.value
      const line = this.token.line
      this.position++
      left = { kind: 'binop', op, left, right: next(), line }
    }
    return left
  }

  private orExpression(): Expr { return this.binaryLevel(['|'], () => this.xorExpression()) }
  private xorExpression(): Expr { return this.binaryLevel(['^'], () => this.andExpression()) }
  private andExpression(): Expr { return this.binaryLevel(['&'], () => this.shiftExpression()) }
  private shiftExpression(): Expr { return this.binaryLevel(['<<', '>>'], () => this.arithmetic()) }
  private arithmetic(): Expr { return this.binaryLevel(['+', '-'], () => this.term()) }
  private term(): Expr { return this.binaryLevel(['*', '/', '//', '%', '@'], () => this.factor()) }

  private factor(): Expr {
    const line = this.token.line
    if (this.isOp('-') || this.isOp('+') || this.isOp('~')) {
      const op = this.token.value
      this.position++
      return { kind: 'unary', op, operand: this.factor(), line }
    }
    return this.power()
  }

  private power(): Expr {
    const line = this.token.line
    if (this.isName('await')) this.position++
    const base = this.primary()
    if (this.accept('**')) return { kind: 'binop', op: '**', left: base, right: this.factor(), line }
    return base
  }

  private primary(): Expr {
    let expression = this.atom()
    while (true) {
      const line = this.token.line
      if (this.accept('.')) expression = { kind: 'attr', object: expression, name: this.expectName(), line }
      else if (this.accept('(')) {
        const args = this.callArguments()
        const close = this.expect(')')
        expression = { kind: 'call', func: expression, args, line: expression.line, endLine: close.line }
      } else if (this.accept('[')) {
        const index = this.subscriptList()
        this.expect(']')
        expression = { kind: 'subscript', object: expression, index, line }
      } else break
    }
    return expression
  }

  private callArguments(): Arg[] {
    const args: Arg[] = []
    while (!this.isOp(')')) {
      if (this.accept('**')) args.push({ value: this.test(), star: '**' })
      else if (this.accept('*')) args.push({ value: this.test(), star: '*' })
      else if (this.token.kind === 'name' && this.peek().kind === 'op' && this.peek().value === '=') {
        const name = this.expectName()
        this.position++
        args.push({ name, value: this.test() })
      } else {
        const value = this.test()
        if (this.isName('for')) args.push({ value: this.comprehension('gen', value, this.token.line) })
        else args.push({ value })
      }
      if (!this.accept(',')) break
    }
    return args
  }

  private subscriptList(): Expr {
    const line = this.token.line
    const first = this.subscriptItem()
    if (!this.isOp(',')) return first
    const items = [first]
    while (this.accept(',')) { if (this.isOp(']')) break; items.push(this.subscriptItem()) }
    return { kind: 'tuple', items, line }
  }

  private subscriptItem(): Expr {
    const line = this.token.line
    let lower: Expr | undefined
    if (!this.isOp(':')) {
      lower = this.test()
      if (!this.isOp(':')) return lower
    }
    this.expect(':')
    const upper = this.isOp(':') || this.isOp(']') || this.isOp(',') ? undefined : this.test()
    let step: Expr | undefined
    if (this.accept(':')) step = this.isOp(']') || this.isOp(',') ? undefined : this.test()
    return { kind: 'slice', lower, upper, step, line }
  }

  private comprehension(type: 'list' | 'gen' | 'set' | 'dict', element: Expr, line: number, value?: Expr): Expr {
    this.expect('for')
    const target = this.targetList()
    this.expect('in')
    const iter = this.orTest()
    const conditions: Expr[] = []
    while (this.accept('if')) conditions.push(this.orTestNoTernary())
    if (this.isName('for')) {
      // Nested comprehensions are rare in model code; flatten by treating the inner loop as the element source.
      const inner = this.comprehension(type, element, line, value)
      if (inner.kind === 'comp') inner.nested = true
      return { kind: 'comp', type, element: inner, target, iter, conditions, line }
    }
    return { kind: 'comp', type, element, value, target, iter, conditions, line }
  }

  private orTestNoTernary(): Expr { return this.orTest() }

  /** Splits an f-string template into literal text and parsed `{…}` fields. */
  private fstringParts(template: string, raw: boolean, line: number): FStringPart[] {
    const parts: FStringPart[] = []
    const literal = (text: string) => { if (text) parts.push(raw ? text : unescapePython(text)) }
    let text = ''
    let index = 0
    while (index < template.length) {
      const char = template[index]
      if ((char === '{' || char === '}') && template[index + 1] === char) { text += char; index += 2; continue }
      if (char === '}') throw new PySyntaxError("f-string: single '}' is not allowed", line, this.file)
      if (char !== '{') { text += char; index++; continue }
      literal(text)
      text = ''
      // Find the end of the field, skipping nested brackets and quoted strings.
      let depth = 0
      let quote = ''
      let cursor = index + 1
      let exprEnd = -1
      let conversion: 'r' | 's' | 'a' | undefined
      let specStart = -1
      for (; cursor < template.length; cursor++) {
        const current = template[cursor]
        if (quote) { if (current === quote) quote = ''; continue }
        if (current === '"' || current === "'") { quote = current; continue }
        if ('([{'.includes(current)) { depth++; continue }
        if (depth > 0 && ')]}'.includes(current)) { depth--; continue }
        if (depth > 0) continue
        if (specStart >= 0) { if (current === '{') depth++; else if (current === '}') break; continue }
        if (current === '!' && template[cursor + 1] !== '=') {
          if (exprEnd < 0) exprEnd = cursor
          const kind = template[cursor + 1]
          if (kind !== 'r' && kind !== 's' && kind !== 'a') throw new PySyntaxError(`f-string: invalid conversion character '${kind}'`, line, this.file)
          conversion = kind
          cursor++
          continue
        }
        if (current === ':') { if (exprEnd < 0) exprEnd = cursor; specStart = cursor + 1; continue }
        if (current === '}') break
      }
      if (cursor >= template.length) throw new PySyntaxError("f-string: expecting '}'", line, this.file)
      if (exprEnd < 0) exprEnd = cursor
      let source = template.slice(index + 1, exprEnd)
      let debug: string | undefined
      const trimmed = source.trimEnd()
      if (trimmed.endsWith('=') && !/[=!<>]=$/.test(trimmed)) { debug = source; source = trimmed.slice(0, -1) }
      if (!source.trim()) throw new PySyntaxError('f-string: empty expression not allowed', line, this.file)
      const parser = new Parser('\n'.repeat(Math.max(0, line - 1)) + '(' + source + ')', this.file)
      const expr = parser.test()
      const spec = specStart >= 0 ? this.fstringParts(template.slice(specStart, cursor), raw, line) : undefined
      parts.push({ expr, conversion, spec, debug })
      index = cursor + 1
    }
    literal(text)
    return parts
  }

  private atom(): Expr {
    const token = this.token
    const line = token.line
    if (token.kind === 'number') {
      this.position++
      const text = token.value
      const value = /^0[xX]/.test(text) ? parseInt(text, 16) : /^0[bB]/.test(text) ? parseInt(text.slice(2), 2) : /^0[oO]/.test(text) ? parseInt(text.slice(2), 8) : Number(text.replace(/j$/, ''))
      return { kind: 'const', value, line }
    }
    if (token.kind === 'string') {
      const parts: FStringPart[] = []
      let formatted = false
      while (this.token.kind === 'string') {
        if (this.token.fstring) { formatted = true; parts.push(...this.fstringParts(this.token.value, Boolean(this.token.raw), this.token.line)) }
        else parts.push(this.token.value)
        this.position++
      }
      if (!formatted) return { kind: 'const', value: parts.join(''), line }
      return { kind: 'fstring', parts: mergeLiterals(parts), line }
    }
    if (token.kind === 'name') {
      this.position++
      if (token.value === 'True') return { kind: 'const', value: true, line }
      if (token.value === 'False') return { kind: 'const', value: false, line }
      if (token.value === 'None') return { kind: 'const', value: null, line }
      return { kind: 'name', id: token.value, line }
    }
    if (this.accept('...')) return { kind: 'ellipsis', line }
    if (this.accept('(')) {
      if (this.accept(')')) return { kind: 'tuple', items: [], line }
      const first = this.starOr(() => this.test())
      if (this.isName('for')) {
        const comp = this.comprehension('gen', first, line)
        this.expect(')')
        return comp
      }
      if (this.accept(')')) return first
      const items = [first]
      while (this.accept(',')) { if (this.isOp(')')) break; items.push(this.starOr(() => this.test())) }
      this.expect(')')
      return { kind: 'tuple', items, line }
    }
    if (this.accept('[')) {
      if (this.accept(']')) return { kind: 'list', items: [], line }
      const first = this.starOr(() => this.test())
      if (this.isName('for')) {
        const comp = this.comprehension('list', first, line)
        this.expect(']')
        return comp
      }
      const items = [first]
      while (this.accept(',')) { if (this.isOp(']')) break; items.push(this.starOr(() => this.test())) }
      this.expect(']')
      return { kind: 'list', items, line }
    }
    if (this.accept('{')) {
      if (this.accept('}')) return { kind: 'dict', keys: [], values: [], line }
      if (this.accept('**')) {
        const keys: (Expr | null)[] = [null]
        const values = [this.test()]
        while (this.accept(',')) {
          if (this.isOp('}')) break
          if (this.accept('**')) { keys.push(null); values.push(this.test()) } else { keys.push(this.test()); this.expect(':'); values.push(this.test()) }
        }
        this.expect('}')
        return { kind: 'dict', keys, values, line }
      }
      const first = this.test()
      if (this.accept(':')) {
        const firstValue = this.test()
        if (this.isName('for')) {
          const comp = this.comprehension('dict', first, line, firstValue)
          this.expect('}')
          return comp
        }
        const keys: (Expr | null)[] = [first]
        const values = [firstValue]
        while (this.accept(',')) {
          if (this.isOp('}')) break
          if (this.accept('**')) { keys.push(null); values.push(this.test()); continue }
          keys.push(this.test())
          this.expect(':')
          values.push(this.test())
        }
        this.expect('}')
        return { kind: 'dict', keys, values, line }
      }
      if (this.isName('for')) {
        const comp = this.comprehension('set', first, line)
        this.expect('}')
        return comp
      }
      const items = [first]
      while (this.accept(',')) { if (this.isOp('}')) break; items.push(this.test()) }
      this.expect('}')
      return { kind: 'set', items, line }
    }
    return this.fail(`Unexpected '${token.value || token.kind}'`)
  }
}

export function parsePython(source: string, file: string): Stmt[] {
  return new Parser(source, file).parseModule()
}

function mergeLiterals(parts: FStringPart[]): FStringPart[] {
  const merged: FStringPart[] = []
  for (const part of parts) {
    if (typeof part === 'string' && typeof merged[merged.length - 1] === 'string') merged[merged.length - 1] += part
    else merged.push(part)
  }
  return merged
}

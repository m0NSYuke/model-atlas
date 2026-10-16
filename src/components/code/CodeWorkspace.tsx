import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import Editor, { loader, type OnMount } from '@monaco-editor/react'
import * as monaco from 'monaco-editor/editor/editor.api'
import 'monaco-editor/languages/definitions/python/register'
import 'monaco-editor/language/json/monaco.contribution'
import EditorWorker from 'monaco-editor/editor/editor.worker?worker'
import JsonWorker from 'monaco-editor/language/json/json.worker?worker'
import { AlertCircle, ArrowRight, Braces, CheckCircle2, Code2, FileCode2, FlaskConical, Ruler, RotateCcw, Table2, Terminal } from 'lucide-react'
import { getNode } from '../../core/model-ir/utils'
import { localizedNode, t, typeName } from '../../i18n/translations'
import { localizeDiagnostic } from '../../i18n/diagnostics'
import { useModelStore } from '../../store/modelStore'
import { codeForStudy, codeModelById } from '../../data/models/codeCatalog'
import type { ModelIR, ModelNode } from '../../core/model-ir/types'
import { formatCount, usePalette } from '../scene/theme'
import { useUiStore } from '../../store/uiStore'
import { tr } from '../scene/useViewGraph'
import { useStudyStore } from '../../store/studyStore'
import { lineChanges } from '../../core/study/lineDiff'
import { CompareBar } from './CompareBar'

Object.assign(self, {
  MonacoEnvironment: { getWorker: (_moduleId: string, label: string) => label === 'json' ? new JsonWorker() : new EditorWorker() },
})
loader.config({ monaco })

let themeReady = false
const IR = 'model-ir.json'
const opTypes = new Set(['add', 'multiply', 'matmul', 'concat', 'split', 'reshape', 'generic', 'parameter', 'tensor'])

function defineTheme() {
  if (themeReady) return
  monaco.editor.defineTheme('atlas-night', {
    base: 'vs-dark', inherit: true,
    rules: [
      { token: 'keyword', foreground: 'C4B5FD' },
      { token: 'string', foreground: '86EFAC' },
      { token: 'number', foreground: 'FCD34D' },
      { token: 'comment', foreground: '5E6A82', fontStyle: 'italic' },
      { token: 'identifier', foreground: 'DCE3F0' },
      { token: 'type', foreground: '7DD3FC' },
      { token: 'delimiter', foreground: '8B95AB' },
    ],
    colors: {
      'editor.background': '#10131C', 'editor.foreground': '#DCE3F0',
      'editorLineNumber.foreground': '#3A4257', 'editorLineNumber.activeForeground': '#A5B0C6',
      'editor.selectionBackground': '#6366F150', 'editor.lineHighlightBackground': '#171B27', 'editor.lineHighlightBorder': '#00000000',
      'editorCursor.foreground': '#A5B4FC', 'editorGutter.background': '#10131C',
      'editorIndentGuide.background1': '#1E2333', 'editorIndentGuide.activeBackground1': '#353C52',
      'scrollbarSlider.background': '#2A30425C', 'scrollbarSlider.hoverBackground': '#3A425A88', 'editorWidget.background': '#161A25',
    },
  })
  monaco.editor.defineTheme('atlas-day', {
    base: 'vs', inherit: true,
    rules: [
      { token: 'keyword', foreground: '7C3AED' },
      { token: 'string', foreground: '15803D' },
      { token: 'number', foreground: 'B45309' },
      { token: 'comment', foreground: '94A3B8', fontStyle: 'italic' },
      { token: 'identifier', foreground: '1E293B' },
      { token: 'type', foreground: '0369A1' },
      { token: 'delimiter', foreground: '64748B' },
    ],
    colors: {
      'editor.background': '#FFFFFF', 'editor.foreground': '#1E293B',
      'editorLineNumber.foreground': '#CBD2DE', 'editorLineNumber.activeForeground': '#64748B',
      'editor.selectionBackground': '#6366F12E', 'editor.lineHighlightBackground': '#F5F6FB', 'editor.lineHighlightBorder': '#00000000',
      'editorCursor.foreground': '#4F46E5', 'editorGutter.background': '#FFFFFF',
      'editorIndentGuide.background1': '#EEF0F5', 'editorIndentGuide.activeBackground1': '#D5DAE5',
      'scrollbarSlider.background': '#94A3B833', 'scrollbarSlider.hoverBackground': '#94A3B866', 'editorWidget.background': '#FFFFFF',
    },
  })
  themeReady = true
}

const monacoTheme = () => useUiStore.getState().resolved === 'dark' ? 'atlas-night' : 'atlas-day'

/** The node a cursor line most likely refers to: a forward() call on that line, else the tightest definition around it. */
function nodeAtLine(model: ModelIR, file: string, line: number): ModelNode | undefined {
  const called = model.nodes.filter((node) => node.callSite?.file === file && node.callSite.line === line && node.type !== 'parameter')
  if (called.length) return called[0]
  const defined = model.nodes.filter((node) => node.source?.file === file && node.source.startLine <= line && node.source.endLine >= line && node.type !== 'parameter')
  return defined.sort((a, b) => (a.source!.endLine - a.source!.startLine) - (b.source!.endLine - b.source!.startLine))[0]
}

/**
 * Shape lens: what each forward() line produces, written at the end of the line. A line's result is
 * the value nothing else on that line consumes (x = self.norm(x + self.attn(x)) → the norm's output).
 */
function shapeLens(model: ModelIR, file: string): Map<number, string> {
  const onLine = new Map<number, ModelNode[]>()
  for (const node of model.nodes) {
    if (node.callSite?.file !== file || node.type === 'parameter' || !node.outputShape) continue
    onLine.set(node.callSite.line, [...(onLine.get(node.callSite.line) ?? []), node])
  }
  const lens = new Map<number, string>()
  for (const [line, all] of onLine) {
    // A module called on this line stands for everything inside it (nn.Sequential children share its line).
    const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
    const onThisLine = new Set(all.map((node) => node.id))
    const nodes = all.filter((node) => { for (let current = parents.get(node.id); current; current = parents.get(current)) if (onThisLine.has(current)) return false; return true })
    const ids = new Set(nodes.map((node) => node.id))
    const consumed = new Set(model.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)).map((edge) => edge.source))
    const results = nodes.filter((node) => !consumed.has(node.id))
    const shapes = [...new Set(results.map((node) => node.outputShape!.join('×')))]
    const variable = results.length && results.every((node) => node.variable === results[0].variable) ? results[0].variable : undefined
    const calls = results.length > 1 ? `  ×${results.length}` : ''
    lens.set(line, `${variable ? `${variable}: ` : ''}${shapes.slice(0, 2).join(' | ')}${shapes.length > 2 ? ' …' : ''}${calls}`)
  }
  return lens
}

function SummaryTable() {
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const selected = useModelStore((state) => state.selectedNodeId)
  const selectNode = useModelStore((state) => state.selectNode)
  const { types: typeColor } = usePalette()
  const rows = useMemo(() => {
    const depth = new Map<string, number>()
    const parents = new Map(model.nodes.map((node) => [node.id, node.parentId]))
    model.nodes.forEach((node) => { let level = 0; for (let current = node.parentId; current; current = parents.get(current)) level++; depth.set(node.id, level) })
    return model.nodes.filter((node) => !opTypes.has(node.type) || node.paramCount).map((node) => ({ node, depth: depth.get(node.id) ?? 0 }))
  }, [model])
  return <div className="summary-table">
    <table>
      <thead><tr><th>{tr(locale, 'Layer (type)', '层（类型）')}</th><th>{tr(locale, 'Output shape', '输出形状')}</th><th>{tr(locale, 'Params', '参数')}</th></tr></thead>
      <tbody>
        {rows.map(({ node, depth }) => <tr key={node.id} className={selected === node.id ? 'selected' : ''} onClick={() => selectNode(node.id)}>
          <td style={{ paddingLeft: 10 + depth * 14 }}><i style={{ background: typeColor[node.type] }} />{localizedNode(node, locale, model.id).name}<small>{node.className ?? typeName(locale, node.type)}</small></td>
          <td><code>{node.outputShape ? `[${node.outputShape.join(', ')}]` : '—'}</code></td>
          <td className="num">{node.paramCount ? node.paramCount.toLocaleString() : '—'}</td>
        </tr>)}
      </tbody>
      {model.totalParams !== undefined && <tfoot><tr><td>{tr(locale, 'Total parameters', '总参数量')}</td><td /><td className="num">{model.totalParams.toLocaleString()} <small>({formatCount(model.totalParams)})</small></td></tr></tfoot>}
    </table>
  </div>
}

export function CodeWorkspace({ onCollapse, collapseIcon }: { onCollapse?: () => void; collapseIcon?: ReactNode }) {
  const locale = useModelStore((state) => state.locale)
  const model = useModelStore((state) => state.modelIR)
  const selectedNodeId = useModelStore((state) => state.selectedNodeId)
  const activeFile = useModelStore((state) => state.activeFile)
  const activeLine = useModelStore((state) => state.activeLine)
  const sourceRevision = useModelStore((state) => state.sourceRevision)
  const codeFiles = useModelStore((state) => state.codeFiles)
  const parseError = useModelStore((state) => state.parseError)
  const diagnostic = useModelStore((state) => state.diagnostic)
  const logs = useModelStore((state) => state.logs)
  const modelOrigin = useModelStore((state) => state.modelOrigin)
  const builtInId = useModelStore((state) => state.builtInId)
  const setActiveFile = useModelStore((state) => state.setActiveFile)
  const updateCode = useModelStore((state) => state.updateCode)
  const applyEdit = useModelStore((state) => state.applyEdit)
  const resetCode = useModelStore((state) => state.resetCode)
  const resolvedTheme = useUiStore((state) => state.resolved)
  const [tab, setTab] = useState<'code' | 'summary'>('code')
  const [consoleOpen, setConsoleOpen] = useState(false)
  const [editorReady, setEditorReady] = useState(0)
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  const decorationsRef = useRef<monaco.editor.IEditorDecorationsCollection | null>(null)
  const lensRef = useRef<monaco.editor.IEditorDecorationsCollection | null>(null)
  const diffRef = useRef<monaco.editor.IEditorDecorationsCollection | null>(null)
  const lensOn = useStudyStore((state) => state.shapeLens)
  const cursorTimer = useRef<ReturnType<typeof setTimeout>>()
  const irJson = useMemo(() => activeFile === IR ? JSON.stringify(model, null, 2) : '', [model, activeFile])
  const selectedNode = getNode(model, selectedNodeId)
  const isIR = activeFile === IR
  const value = isIR ? irJson : codeFiles[activeFile] ?? ''
  const entry = builtInId ? codeModelById[builtInId] : undefined
  const pristine = entry ? Object.entries(entry.files).every(([file, source]) => codeFiles[file] === source) : true

  const mount = useCallback<OnMount>((editor) => {
    editorRef.current = editor
    decorationsRef.current = editor.createDecorationsCollection()
    lensRef.current = editor.createDecorationsCollection()
    diffRef.current = editor.createDecorationsCollection()
    setEditorReady((count) => count + 1)
    defineTheme()
    monaco.editor.setTheme(monacoTheme())
    editor.onDidChangeCursorPosition((event) => {
      if (event.source !== 'mouse' && event.source !== 'keyboard') return
      clearTimeout(cursorTimer.current)
      cursorTimer.current = setTimeout(() => {
        const state = useModelStore.getState()
        if (state.modelOrigin !== 'code' || state.activeFile === IR) return
        const node = nodeAtLine(state.modelIR, state.activeFile, event.position.lineNumber)
        if (node && node.id !== state.selectedNodeId) state.selectNode(node.id, { fromEditor: true })
      }, 160)
    })
  }, [])

  useEffect(() => { if (themeReady) monaco.editor.setTheme(monacoTheme()) }, [resolvedTheme])

  // Highlight the selected node's definition and the line where forward() calls it.
  useEffect(() => {
    const editor = editorRef.current
    if (!editor || !decorationsRef.current) return
    const decorations: monaco.editor.IModelDeltaDecoration[] = []
    const source = selectedNode?.source
    if (source && source.file === activeFile) {
      const endLine = Math.min(source.endLine, editor.getModel()?.getLineCount() ?? source.endLine)
      decorations.push({ range: new monaco.Range(source.startLine, 1, endLine, 1), options: { isWholeLine: true, className: 'source-range-highlight', linesDecorationsClassName: 'source-line-gutter' } })
    }
    const call = selectedNode?.callSite
    if (call && call.file === activeFile) decorations.push({ range: new monaco.Range(call.line, 1, call.line, 1), options: { isWholeLine: true, className: 'call-line-highlight', linesDecorationsClassName: 'call-line-gutter' } })
    decorationsRef.current.set(decorations)
  }, [activeFile, selectedNode, value])

  // Shape lens: the traced output shape at the end of every forward() line.
  useEffect(() => {
    const lenses = lensRef.current
    if (!lenses) return
    if (!lensOn || isIR || modelOrigin !== 'code') { lenses.set([]); return }
    const editorModel = editorRef.current?.getModel()
    if (!editorModel) return
    const lineCount = editorModel.getLineCount()
    lenses.set([...shapeLens(model, activeFile)].filter(([line]) => line <= lineCount).map(([line, text]) => {
      const end = editorModel.getLineMaxColumn(line)
      return { range: new monaco.Range(line, end, line, end), options: { after: { content: `   ⟶ ${text}`, inlineClassName: 'shape-lens' }, showIfCollapsed: true } }
    }))
  }, [model, activeFile, lensOn, isIR, modelOrigin, value, editorReady])

  // Edited lines compared with the original file, like a version-control gutter.
  useEffect(() => {
    const marks = diffRef.current
    if (!marks) return
    const original = entry?.files[activeFile]
    if (isIR || original === undefined || original === value) { marks.set([]); return }
    const lineCount = editorRef.current?.getModel()?.getLineCount() ?? 1
    const { added, modified, deleted } = lineChanges(original, value)
    const line = (number: number, gutter: string, row?: string): monaco.editor.IModelDeltaDecoration => ({ range: new monaco.Range(Math.max(1, Math.min(lineCount, number)), 1, Math.max(1, Math.min(lineCount, number)), 1), options: { isWholeLine: true, linesDecorationsClassName: gutter, className: row } })
    marks.set([...added.map((number) => line(number, 'diff-gutter added', 'diff-row added')), ...modified.map((number) => line(number, 'diff-gutter modified', 'diff-row modified')), ...deleted.map((number) => line(Math.max(1, number), 'diff-gutter deleted'))])
  }, [entry, activeFile, value, isIR, editorReady])

  useEffect(() => {
    const editor = editorRef.current
    if (!editor || activeLine === null) return
    editor.revealLineInCenterIfOutsideViewport(activeLine)
    if (!editor.hasTextFocus()) editor.setPosition({ lineNumber: activeLine, column: 1 })
  }, [activeLine, sourceRevision])

  // Error squiggles from the tracer.
  useEffect(() => {
    const editor = editorRef.current
    const editorModel = editor?.getModel()
    if (!editorModel) return
    const markers: monaco.editor.IMarkerData[] = diagnostic?.line && (!diagnostic.file || diagnostic.file === activeFile) ? [{
      severity: monaco.MarkerSeverity.Error, message: diagnostic.message,
      startLineNumber: diagnostic.line, endLineNumber: diagnostic.line, startColumn: 1, endColumn: editorModel.getLineMaxColumn(Math.min(diagnostic.line, editorModel.getLineCount())),
    }] : []
    monaco.editor.setModelMarkers(editorModel, 'atlas', markers)
  }, [diagnostic, activeFile, value])

  const files = Object.keys(codeFiles)
  const isCode = modelOrigin === 'code'
  const hint = isCode ? tr(locale, 'Edit the PyTorch code — the 3D graph is rebuilt from it as you type', '修改 PyTorch 代码，3D 结构会随输入自动重建') : t(locale, modelOrigin === 'demo' ? 'codeHint' : modelOrigin === 'catalog' ? 'catalogCodeHint' : 'importedCodeHint')

  return <section className="panel code-panel">
    <div className="code-tabs">
      <div className="code-tab-scroll">
      {files.map((file) => <button type="button" className={`code-tab ${tab === 'code' && activeFile === file ? 'active' : ''}`} onClick={() => { setTab('code'); setActiveFile(file) }} key={file}><FileCode2 size={13} />{file}</button>)}
      <button type="button" className={`code-tab ${tab === 'code' && isIR ? 'active' : ''}`} onClick={() => { setTab('code'); setActiveFile(IR) }}><Braces size={13} />{t(locale, 'irTab')}</button>
      {isCode && <button type="button" className={`code-tab ${tab === 'summary' ? 'active' : ''}`} onClick={() => setTab('summary')}><Table2 size={13} />{tr(locale, 'Summary', '参数表')}</button>}
      </div>
      <span className="code-tabs-spacer" />
      {(isCode || modelOrigin === 'demo') && <button type="button" className="code-reset" disabled={pristine && modelOrigin === 'code'} onClick={() => { if (modelOrigin !== 'code' || window.confirm(tr(locale, 'Restore the original code? Your edits to this model will be discarded.', '恢复原始代码吗？你对这个模型的修改会被丢弃。'))) resetCode() }} title={tr(locale, 'Restore the original code', '恢复原始代码')}><RotateCcw size={13} />{tr(locale, 'Reset', '重置')}</button>}
      {onCollapse && <button type="button" className="icon-button small" onClick={onCollapse} data-tip={tr(locale, 'Hide code  [', '收起代码栏  [')} data-tip-side="left" aria-label={tr(locale, 'Hide code', '收起代码栏')}>{collapseIcon}</button>}
    </div>

    {modelOrigin === 'catalog' && builtInId && codeForStudy[builtInId] && <button type="button" className="study-code-link" onClick={() => useModelStore.getState().loadBuiltIn(codeForStudy[builtInId])}>
      <Code2 size={14} /><span><b>{tr(locale, 'Runnable PyTorch version', '可运行的 PyTorch 版本')}</b>{tr(locale, 'Open the code model this map summarizes: edit it and watch the graph rebuild.', '打开这张图谱对应的代码模型：可以修改代码，结构图会随之重建。')}</span><ArrowRight size={14} />
    </button>}

    {entry && entry.experiments.length > 0 && tab === 'code' && <div className="experiments">
      <span className="experiments-label"><FlaskConical size={13} />{tr(locale, 'Experiments', '实验')}</span>
      <div className="experiments-list">
        {entry.experiments.map((experiment) => {
          const file = experiment.file ?? 'model.py'
          const available = codeFiles[file]?.includes(experiment.find) ?? false
          return <button type="button" key={experiment.label.en} disabled={!available} title={`${experiment.hint[locale]}${available ? '' : tr(locale, ' (already applied, or needs another experiment first)', '（已应用，或需先做另一个实验）')}`}
            onClick={() => { setTab('code'); applyEdit(file, experiment.find, experiment.replace) }}>{experiment.label[locale]}</button>
        })}
      </div>
    </div>}

    {isCode && tab === 'code' && <CompareBar />}

    <div className="monaco-wrap">
      {tab === 'summary' ? <SummaryTable /> : <Editor path={activeFile} language={isIR ? 'json' : activeFile.endsWith('.py') ? 'python' : 'plaintext'} value={value} onChange={(next) => {
        const current = useModelStore.getState()
        if (!isIR && current.activeFile === activeFile && activeFile in current.codeFiles) updateCode(activeFile, next ?? '')
      }} beforeMount={defineTheme} theme={resolvedTheme === 'dark' ? 'atlas-night' : 'atlas-day'} onMount={mount} loading={<div className="editor-loading">{t(locale, 'loadingEditor')}</div>}
        options={{ readOnly: isIR, minimap: { enabled: false }, fontSize: 13, lineHeight: 21, fontFamily: '"JetBrains Mono", "Cascadia Code", "Cascadia Mono", Consolas, monospace', fontLigatures: true, folding: true, lineNumbers: 'on', scrollBeyondLastLine: false, automaticLayout: true, renderLineHighlight: 'line', tabSize: 4, wordWrap: 'off', padding: { top: 12, bottom: 12 }, smoothScrolling: true, cursorSmoothCaretAnimation: 'on', glyphMargin: false, lineDecorationsWidth: 8 }} />}
    </div>

    {consoleOpen && logs.length > 0 && <pre className="code-console">{logs.join('\n')}</pre>}
    <div className={`code-status ${parseError ? 'has-error' : ''}`}>
      {parseError ? <AlertCircle size={13} /> : <CheckCircle2 size={13} />}
      <span className="code-status-text">{parseError
        ? `${diagnostic?.line ? tr(locale, `Line ${diagnostic.line}: `, `第 ${diagnostic.line} 行：`) : ''}${localizeDiagnostic(parseError, locale)}`
        : isCode ? tr(locale, `Traced ${model.nodes.length} nodes · ${model.edges.length} edges`, `已追踪 ${model.nodes.length} 个节点 · ${model.edges.length} 条连线`) : hint}</span>
      <span className="code-status-spacer" />
      {isCode && !isIR && <button type="button" className={lensOn ? 'on' : ''} onClick={() => useStudyStore.getState().setShapeLens(!lensOn)} title={tr(locale, 'Show the traced output shape at the end of each forward() line', '在 forward() 每行末尾显示追踪到的输出形状')}><Ruler size={12} />{tr(locale, 'Shapes', '形状')}</button>}
      {logs.length > 0 && <button type="button" className={consoleOpen ? 'on' : ''} onClick={() => setConsoleOpen((open) => !open)}><Terminal size={12} />print ({logs.length})</button>}
      <span>{isIR ? 'JSON' : activeFile.endsWith('.py') ? 'Python' : 'Text'}</span>
    </div>
  </section>
}

import { useMemo, useRef, useState, type DragEvent } from 'react'
import { Check, ClipboardCopy, FileCode2, FileJson2, FolderOpen, LayoutTemplate, Play, Upload, X } from 'lucide-react'
import { importModel } from '../../core/model-ir/importModel'
import { buildModelFromCode, CodeBuildError, executionOrder, isTrainingScript, projectConfig } from '../../core/pytorch'
import { notesKey, useStudyStore } from '../../store/studyStore'
import { t } from '../../i18n/translations'
import { localizeDiagnostic } from '../../i18n/diagnostics'
import { useModelStore } from '../../store/modelStore'
import { formatCount } from '../scene/theme'
import aiPrompt from '../../../templates/ai-import-prompt.md?raw'

const MAX_FILE = 2_000_000
const MAX_TOTAL = 6_000_000
const MAX_FILES = 60
/** Folders that never hold model code. */
const skipped = /(^|\/)(node_modules|\.git|__pycache__|\.venv|venv|\.ipynb_checkpoints|wandb|checkpoints?)\//

interface Picked { name: string; source: string }

/**
 * Import is local: files are read in the browser and traced here. A project is any set of .py files (plus an
 * optional atlas.json); the reader picks which files to include and which one to run, as `python entry.py`.
 */
export function ImportModelDialog({ onClose }: { onClose: () => void }) {
  const locale = useModelStore((state) => state.locale)
  const apply = useModelStore((state) => state.importProject)
  const importFiles = useModelStore((state) => state.importFiles)
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [reading, setReading] = useState(false)
  const [picked, setPicked] = useState<Picked[]>([])
  const [folder, setFolder] = useState<string | null>(null)
  const [included, setIncluded] = useState<Set<string>>(new Set())
  const [entry, setEntry] = useState<string>('')
  const [copied, setCopied] = useState(false)
  const zh = locale === 'zh'

  async function receive(list: File[]) {
    setError(null)
    const files = list.filter((file) => !skipped.test((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name))
    if (!files.length) return
    // A single Model IR JSON or exported project keeps the direct route.
    if (files.length === 1 && files[0].name.endsWith('.json') && files[0].name !== 'atlas.json') { await receiveJson(files[0]); return }
    const code = files.filter((file) => file.name.endsWith('.py') || file.name === 'atlas.json')
    if (!code.length) { setError(zh ? '没有找到 .py 文件' : 'No .py files found'); return }
    if (code.length > MAX_FILES) { setError(zh ? `文件太多（${code.length} 个），请只选模型相关的文件夹` : `Too many files (${code.length}); pick the folder with the model code`); return }
    const big = code.find((file) => file.size > MAX_FILE)
    if (big) { setError(zh ? `${big.name} 超过 2 MB` : `${big.name} is larger than 2 MB`); return }
    if (code.reduce((sum, file) => sum + file.size, 0) > MAX_TOTAL) { setError(zh ? '文件总大小超过 6 MB' : 'Files exceed 6 MB in total'); return }
    setReading(true)
    try {
      const relative = (file: File) => ((file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name)
      const root = relative(code[0]).includes('/') ? relative(code[0]).split('/')[0] : null
      // Files keep their path inside the chosen folder; nested folders flatten to their file name when unique.
      const names = code.map((file) => relative(file).split('/').slice(root ? 1 : 0).join('/'))
      const flat = names.map((name) => name.split('/').pop()!)
      const unique = new Set(flat).size === flat.length
      const read = await Promise.all(code.map(async (file, index) => ({ name: unique ? flat[index] : names[index], source: await file.text() })))
      read.sort((a, b) => order(a.name) - order(b.name) || a.name.localeCompare(b.name))
      const files = Object.fromEntries(read.map((file) => [file.name, file.source]))
      setPicked(read)
      setFolder(root)
      setIncluded(new Set(read.map((file) => file.name)))
      setEntry(executionOrder(files).at(-1) ?? '')
    } finally {
      setReading(false)
    }
  }

  async function receiveJson(file: File) {
    if (file.size > MAX_FILE) { setError(zh ? '文件不能超过 2 MB' : 'File must be smaller than 2 MB'); return }
    setReading(true)
    try {
      const json = JSON.parse(await file.text())
      const project = importModel(json)
      apply(project)
      // Notes exported with the project come back with it.
      const notes = json?.study?.notes
      if (notes && typeof notes === 'object') useStudyStore.getState().mergeNotes(notesKey(useModelStore.getState()), Object.fromEntries(Object.entries(notes).filter(([, text]) => typeof text === 'string')) as Record<string, string>)
      onClose()
    } catch (issue) {
      setError(localizeDiagnostic(issue instanceof Error ? issue.message : 'Could not read this JSON file', locale))
    } finally {
      setReading(false)
    }
  }

  /** The files to import: the ticked ones, with the chosen entry recorded in atlas.json when it is not the default. */
  const project = useMemo(() => {
    const files: Record<string, string> = Object.fromEntries(picked.filter((file) => included.has(file.name)).map((file) => [file.name, file.source]))
    if (entry && files[entry] !== undefined && executionOrder(files).at(-1) !== entry) {
      let config: Record<string, unknown> = {}
      try { config = JSON.parse(files['atlas.json'] ?? '{}') } catch { /* replaced below */ }
      files['atlas.json'] = JSON.stringify({ ...config, entry }, null, 2) + '\n'
    }
    return files
  }, [picked, included, entry])

  const preview = useMemo(() => {
    if (!Object.keys(project).some((file) => file.endsWith('.py'))) return null
    try {
      const { model } = buildModelFromCode('preview', project)
      const training = model.nodes.some((node) => node.type === 'optimizer')
      return { ok: true as const, params: model.totalParams ?? 0, blocks: model.nodes.length, training, order: executionOrder(project) }
    } catch (issue) {
      const diagnostic = issue instanceof CodeBuildError ? issue.diagnostic : { message: issue instanceof Error ? issue.message : String(issue) }
      return { ok: false as const, message: localizeDiagnostic(diagnostic.message, locale), where: diagnostic.file ? `${diagnostic.file}${diagnostic.line ? `:${diagnostic.line}` : ''}` : '', order: executionOrder(project) }
    }
  }, [project, locale])

  function confirm() {
    const config = projectConfig(project)
    const name = config.name ?? folder ?? (entry.replace(/\.py$/, '') || 'my-model')
    importFiles(name, project)
    onClose()
  }

  function toggle(name: string) {
    setIncluded((current) => {
      const next = new Set(current)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  async function copyPrompt() {
    try {
      await navigator.clipboard.writeText(aiPrompt)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      setError(zh ? '无法写入剪贴板，请打开项目里的 templates/ai-import-prompt.md 复制' : 'Clipboard unavailable; copy templates/ai-import-prompt.md from the project instead')
    }
  }

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    receive([...event.dataTransfer.files])
  }

  const python = picked.filter((file) => file.name.endsWith('.py') && included.has(file.name))

  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="import-dialog" role="dialog" aria-modal="true" aria-labelledby="import-title">
      <div className="dialog-heading"><div className="dialog-icon"><FileJson2 size={19} /></div><div><h2 id="import-title">{t(locale, 'importTitle')}</h2><p>{zh ? '文件只在本机浏览器里读取和追踪，不上传、不联网、不需要 API。' : 'Files are read and traced in this browser only: nothing is uploaded, no network, no API key.'}</p></div><button type="button" className="icon-button" aria-label={t(locale, 'close')} onClick={onClose}><X size={18} /></button></div>

      {!picked.length ? <>
        <div className="import-dropzone" onDragOver={(event) => event.preventDefault()} onDrop={drop}>
          <Upload size={23} />
          <div className="import-pick">
            <button type="button" onClick={() => filesInput.current?.click()} disabled={reading}><FileCode2 size={14} />{reading ? t(locale, 'reading') : zh ? '选择文件' : 'Choose files'}</button>
            <button type="button" className="secondary" onClick={() => folderInput.current?.click()} disabled={reading}><FolderOpen size={14} />{zh ? '选择文件夹' : 'Choose folder'}</button>
          </div>
          <span>{zh ? '可多选 .py 文件（含 atlas.json），或拖拽到这里' : 'Pick several .py files (and atlas.json), or drop them here'}</span>
          <input ref={filesInput} type="file" multiple accept=".py,.json,application/json" onChange={(event) => { receive([...(event.target.files ?? [])]); event.target.value = '' }} hidden />
          <input ref={folderInput} type="file" multiple {...{ webkitdirectory: '' }} onChange={(event) => { receive([...(event.target.files ?? [])]); event.target.value = '' }} hidden />
        </div>
        <div className="import-ai">
          <div>
            <b>{zh ? '代码不符合要求？让 AI 按模板改写' : 'Code not in shape? Let an AI rewrite it to the template'}</b>
            <p>{zh ? '复制提示词，连同你的代码一起发给常用的 AI 助手，它会输出 model.py / data.py / train.py / atlas.json，再导入即可。' : 'Copy the prompt, send it with your code to your AI assistant, and import the model.py / data.py / train.py / atlas.json it returns.'}</p>
          </div>
          <div className="import-ai-actions">
            <button type="button" onClick={copyPrompt}>{copied ? <Check size={13} /> : <ClipboardCopy size={13} />}{copied ? (zh ? '已复制' : 'Copied') : (zh ? '复制 AI 提示词' : 'Copy AI prompt')}</button>
            <button type="button" onClick={() => { useModelStore.getState().loadBuiltIn('code-template'); onClose() }}><LayoutTemplate size={13} />{zh ? '打开模板' : 'Open template'}</button>
          </div>
        </div>
        <p className="import-support">{zh ? '也支持单个 Model IR JSON 或导出的 Atlas 项目。' : 'A single Model IR JSON or an exported Atlas project also works.'}</p>
      </> : <>
        <div className="import-files">
          <div className="import-files-head"><span>{folder ? `${folder}/` : zh ? '已选文件' : 'Selected files'}</span><em>{zh ? '勾选要包含的文件' : 'Tick the files to include'}</em></div>
          {picked.map((file) => {
            const isPython = file.name.endsWith('.py')
            const role = file.name === 'atlas.json' ? (zh ? '项目配置' : 'project config') : isTrainingScript(file.source) ? (zh ? '训练脚本' : 'training') : /nn\.Module/.test(file.source) ? (zh ? '模型' : 'model') : ''
            return <label key={file.name} className={included.has(file.name) ? '' : 'off'}>
              <input type="checkbox" checked={included.has(file.name)} onChange={() => toggle(file.name)} />
              <code>{file.name}</code>
              {role && <small>{role}</small>}
              {isPython && entry === file.name && <b>{zh ? '入口' : 'entry'}</b>}
            </label>
          })}
        </div>
        {python.length > 0 && <label className="import-entry">
          <span>{zh ? '从哪个文件开始运行' : 'Run from'}</span>
          <select value={python.some((file) => file.name === entry) ? entry : ''} onChange={(event) => setEntry(event.target.value)}>
            {!python.some((file) => file.name === entry) && <option value="">{zh ? '自动' : 'Automatic'}</option>}
            {python.map((file) => <option key={file.name} value={file.name}>{file.name}</option>)}
          </select>
          <em>{zh ? '选训练脚本会画出一个训练步，选 model.py 只画网络' : 'A training script draws one training step; model.py draws the network only'}</em>
        </label>}
        {preview && (preview.ok
          ? <div className="import-preview ok"><Check size={14} /><span>{zh ? `可以导入：${preview.blocks} 个节点 · ${formatCount(preview.params)} 参数${preview.training ? ' · 含训练步' : ''}` : `Ready: ${preview.blocks} nodes · ${formatCount(preview.params)} params${preview.training ? ' · with a training step' : ''}`}</span><small>{preview.order.join(' → ')}</small></div>
          : <div className="import-preview bad"><X size={14} /><span>{preview.where && <code>{preview.where}</code>} {preview.message}</span><small>{zh ? '仍可导入，再在编辑器里修改；或用 AI 提示词让它按模板改写。' : 'You can still import and fix it in the editor, or have an AI rewrite it with the prompt.'}</small></div>)}
      </>}
      {error && <div className="import-error" role="alert">{t(locale, 'importError')}: {error}</div>}
      <div className="dialog-actions">
        {picked.length > 0 && <button type="button" onClick={() => { setPicked([]); setError(null) }}>{zh ? '重新选择' : 'Choose again'}</button>}
        <button type="button" onClick={onClose}>{t(locale, 'importCancel')}</button>
        {picked.length > 0 && <button type="button" className="primary" disabled={!python.length} onClick={confirm}><Play size={13} />{zh ? '导入并 3D 化' : 'Import'}</button>}
      </div>
    </div>
  </div>
}

/** model.py first, then other files, training scripts and atlas.json last. */
function order(name: string): number {
  if (name === 'model.py') return 0
  if (name === 'atlas.json') return 3
  if (/train/.test(name)) return 2
  return 1
}

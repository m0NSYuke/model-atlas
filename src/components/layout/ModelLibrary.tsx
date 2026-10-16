import { useEffect, useMemo } from 'react'
import { ArrowUpRight, Code2, FilePlus2, FolderOpen, Map as MapIcon, Trash2, Upload, X } from 'lucide-react'
import { codeModels } from '../../data/models/codeCatalog'
import { studyCatalog, type StudyCategory } from '../../data/models/studyCatalog'
import { buildModelFromCode } from '../../core/pytorch'
import { localizedModel, localizedModelDescription, t } from '../../i18n/translations'
import { useModelStore } from '../../store/modelStore'
import { formatCount, usePalette } from '../scene/theme'
import { tr } from '../scene/useViewGraph'

const categories: { id: StudyCategory; key: 'piCategory' | 'diffusionCategory' | 'videoCategory' }[] = [
  { id: 'pi', key: 'piCategory' }, { id: 'diffusion', key: 'diffusionCategory' }, { id: 'video', key: 'videoCategory' },
]

let summaryCache: Record<string, { params?: number; types: string[] }> | null = null

/** Parameter counts and a color strip of layer kinds for every built-in code model (traced once). */
function codeSummaries() {
  if (summaryCache) return summaryCache
  summaryCache = {}
  for (const entry of codeModels) {
    try {
      const { model } = buildModelFromCode(entry.id, entry.files)
      const groups = new Set(model.nodes.map((node) => node.parentId).filter(Boolean))
      const types = model.nodes.filter((node) => !groups.has(node.id) && node.type !== 'parameter' && node.type !== 'tensor').map((node) => node.type)
      summaryCache[entry.id] = { params: model.totalParams, types }
    } catch { summaryCache[entry.id] = { types: [] } }
  }
  return summaryCache
}

function TypeStrip({ types }: { types: string[] }) {
  const typeColor = usePalette().types as Record<string, string>
  const step = Math.max(1, Math.ceil(types.length / 48))
  return <div className="type-strip">{types.filter((_, index) => index % step === 0).map((type, index) => <i key={index} style={{ background: typeColor[type] }} />)}</div>
}

export function ModelLibrary({ onClose, onImport }: { onClose: () => void; onImport: () => void }) {
  const locale = useModelStore((state) => state.locale)
  const builtInId = useModelStore((state) => state.builtInId)
  const loadBuiltIn = useModelStore((state) => state.loadBuiltIn)
  const loadBlank = useModelStore((state) => state.loadBlank)
  const projects = useModelStore((state) => state.projects)
  const projectKey = useModelStore((state) => state.projectKey)
  const saveFailed = useModelStore((state) => state.saveFailed)
  const edited = new Set(projects.filter((item) => item.key.startsWith('builtin:')).map((item) => item.builtInId))
  const summaries = useMemo(codeSummaries, [])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const open = (id: string) => { loadBuiltIn(id); onClose() }
  const openProject = (key: string) => { useModelStore.getState().openProject(key); onClose() }
  const removeProject = (key: string, name: string) => {
    if (window.confirm(tr(locale, `Delete "${name}" from this browser? This cannot be undone; export it first if you want a copy.`, `从浏览器里删除“${name}”吗？删除后无法恢复，需要留一份请先导出。`))) useModelStore.getState().deleteProject(key)
  }
  const when = (time: number) => new Date(time).toLocaleString(locale === 'zh' ? 'zh-CN' : 'en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

  return <div className="dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="library" role="dialog" aria-modal="true" aria-labelledby="library-title">
      <header className="library-head">
        <div>
          <h2 id="library-title">{tr(locale, 'Model library', '模型库')}</h2>
          <p>{tr(locale, 'Code models are real PyTorch: edit any line and the 3D architecture is rebuilt from it. Study maps are hand-drawn overviews of larger systems.', '代码模型是真实的 PyTorch 代码：改任何一行，3D 结构都会据此重建。学习图谱是对大型系统的手绘概览。')}</p>
        </div>
        <button type="button" className="icon-button" aria-label={t(locale, 'close')} onClick={onClose}><X size={18} /></button>
      </header>

      <div className="library-body">
        {(projects.length > 0 || saveFailed) && <section>
          <h3><FolderOpen size={15} />{tr(locale, 'My projects', '我的项目')}</h3>
          {saveFailed && <p className="library-warning">{tr(locale, 'This browser refused the last save (storage full or disabled). Export your project to keep a copy.', '浏览器拒绝了最近一次保存（存储已满或被禁用）。请导出项目留一份副本。')}</p>}
          <div className="library-list">
            {projects.map((item) => <div key={item.key} className={`library-row project-row ${projectKey === item.key ? 'current' : ''}`}>
              <button type="button" className="project-open" onClick={() => openProject(item.key)}>
                <strong>{item.name}</strong>
                <span>{item.key.startsWith('builtin:') ? tr(locale, 'Your edits to a built-in model', '你修改过的内置模型') : tr(locale, 'Your project', '你的项目')} · {item.fileCount} {tr(locale, item.fileCount === 1 ? 'file' : 'files', '个文件')} · {tr(locale, 'saved', '保存于')} {when(item.updatedAt)}</span>
              </button>
              <button type="button" className="icon-button project-delete" aria-label={tr(locale, `Delete ${item.name}`, `删除 ${item.name}`)} title={tr(locale, 'Delete from this browser', '从浏览器删除')} onClick={() => removeProject(item.key, item.name)}><Trash2 size={14} /></button>
            </div>)}
          </div>
        </section>}

        <section>
          <h3><Code2 size={15} />{tr(locale, 'Editable PyTorch models', '可编辑的 PyTorch 模型')}</h3>
          <div className="library-grid">
            {codeModels.map((entry) => {
              const summary = summaries[entry.id]
              return <button type="button" key={entry.id} className={`library-card ${builtInId === entry.id ? 'current' : ''}`} onClick={() => open(entry.id)}>
                <div className="library-card-top"><span className="chip">{entry.family[locale]}</span>{edited.has(entry.id) ? <span className="edited">{tr(locale, 'Edited', '已修改')}</span> : entry.year && <span className="year">{entry.year}</span>}</div>
                <strong>{locale === 'zh' ? entry.zhName : entry.name}</strong>
                <p>{locale === 'zh' ? entry.zhDescription : entry.description}</p>
                <TypeStrip types={summary?.types ?? []} />
                <div className="library-card-foot">
                  <span>{formatCount(summary?.params)} {tr(locale, 'params', '参数')}</span>
                  <span>{entry.experiments.length} {tr(locale, 'experiments', '个实验')}</span>
                </div>
              </button>
            })}
            <button type="button" className="library-card blank" onClick={() => { loadBlank(); onClose() }}>
              <FilePlus2 size={22} />
              <strong>{tr(locale, 'Blank model', '从空白开始')}</strong>
              <p>{tr(locale, 'Start from a tiny template and write your own nn.Module.', '从一个极简模板开始，编写你自己的 nn.Module。')}</p>
            </button>
            <button type="button" className="library-card blank" onClick={onImport}>
              <Upload size={22} />
              <strong>{tr(locale, 'Import .py / JSON', '导入 .py / JSON')}</strong>
              <p>{tr(locale, 'Load a PyTorch file from disk, or a Model IR JSON export.', '从本地加载 PyTorch 文件或导出的 Model IR JSON。')}</p>
            </button>
          </div>
        </section>

        {categories.map((category) => <section key={category.id}>
          <h3><MapIcon size={15} />{t(locale, category.key)}</h3>
          <div className="library-list">
            {studyCatalog.filter((entry) => entry.category === category.id).map((entry) => <button type="button" key={entry.model.id} className={`library-row ${builtInId === entry.model.id ? 'current' : ''}`} onClick={() => open(entry.model.id)}>
              <strong>{localizedModel(entry.model, locale)}</strong>
              <span>{localizedModelDescription(entry.model, locale)}</span>
              <small>{entry.model.id === 'vae3d' ? tr(locale, 'Editable demo', '可编辑示例') : t(locale, entry.kind === 'architecture' ? 'architectureType' : entry.kind === 'method' ? 'methodType' : 'overviewType')}<ArrowUpRight size={12} /></small>
            </button>)}
          </div>
        </section>)}
      </div>
    </div>
  </div>
}

import { useMemo, useState } from 'react'
import { BookOpenCheck, Check, ExternalLink, FlaskConical, Lightbulb, Map as MapIcon, Route, X } from 'lucide-react'
import { useModelStore } from '../../store/modelStore'
import { notesKey, useStudyStore } from '../../store/studyStore'
import { codeModelById } from '../../data/models/codeCatalog'
import { studyById } from '../../data/models/studyCatalog'
import { localizedNode } from '../../i18n/translations'
import type { ModelNode } from '../../core/model-ir/types'
import { formatCount, usePalette } from '../scene/theme'
import { tr } from '../scene/useViewGraph'

// Reading a model the way researchers read a paper:
//  • S. Keshav, "How to Read a Paper": three passes, each deeper than the last.
//  • Li Mu (李沐): title/abstract/figures first, then the method against prior work, then redo it in your head.
//  • Andrew Ng: four questions to answer after reading; skim many, read few in depth.
//  • Andrej Karpathy: change one thing at a time and predict the result before you run it.

type Prediction = 'more' | 'fewer' | 'same' | 'error'

const questions = [
  { key: 'goal', en: 'What did the authors try to accomplish?', zh: '作者想解决什么问题？' },
  { key: 'key', en: 'What are the key elements of the approach?', zh: '方法里最关键的要素是什么？' },
  { key: 'use', en: 'What can I use myself?', zh: '哪些东西我自己能用上？' },
  { key: 'refs', en: 'Which references do I want to follow?', zh: '还想顺着读哪些参考文献？' },
]

const firstPassChecks = [
  { key: 'p1-problem', en: 'I can say which problem it solves', zh: '能说出它解决什么问题' },
  { key: 'p1-io', en: 'I know what goes in and what comes out', zh: '知道输入是什么、输出是什么' },
  { key: 'p1-core', en: 'I can point at its one key block', zh: '能指出最关键的那个模块' },
]

function topLevel(nodes: ModelNode[]): ModelNode[] {
  return nodes.filter((node) => !node.parentId && node.type !== 'parameter' && node.className !== 'Input' && node.className !== 'Output' && node.className !== undefined)
}

export function ReadingGuide() {
  const model = useModelStore((state) => state.modelIR)
  const locale = useModelStore((state) => state.locale)
  const builtInId = useModelStore((state) => state.builtInId)
  const origin = useModelStore((state) => state.modelOrigin)
  const diagnostic = useModelStore((state) => state.diagnostic)
  const selectNode = useModelStore((state) => state.selectNode)
  const key = useModelStore(notesKey)
  const answers = useStudyStore((state) => state.answers[key])
  const checks = useStudyStore((state) => state.checks[key])
  const { types } = usePalette()
  const [pass, setPass] = useState<1 | 2 | 3>(1)
  const [results, setResults] = useState<Record<string, Prediction>>({})
  const entry = builtInId ? codeModelById[builtInId] : undefined
  const study = builtInId ? studyById[builtInId] ?? (entry?.study ? studyById[entry.study] : undefined) : undefined
  const reference = entry?.reference ?? study?.reference
  const name = (node: ModelNode) => localizedNode(node, locale, model.id).name

  const facts = useMemo(() => {
    const inputs = model.nodes.filter((node) => node.className === 'Input')
    const outputs = model.nodes.filter((node) => node.className === 'Output')
    // Study maps are already the paper's figure: every block is a main block.
    const blocks = model.layout === 'code' ? topLevel(model.nodes).filter((node) => node.paramCount || model.nodes.some((child) => child.parentId === node.id))
      : model.nodes.filter((node) => node.type !== 'tensor' && node.type !== 'latent')
    return { inputs, outputs, blocks }
  }, [model])
  const done = (item: string) => checks?.includes(item) ?? false
  const study4 = questions.filter((question) => answers?.[question.key]?.trim()).length
  const progress = [firstPassChecks.every((item) => done(item.key)), facts.blocks.length > 0 && facts.blocks.every((node) => done(`p2:${node.id}`)), study4 === 4]

  const runExperiment = (label: string, file: string, find: string, replace: string) => {
    const before = useModelStore.getState().modelIR.totalParams ?? 0
    const applied = useModelStore.getState().applyEdit(file, find, replace)
    if (!applied) return
    const after = useModelStore.getState()
    const outcome: Prediction = after.diagnostic ? 'error' : (after.modelIR.totalParams ?? 0) > before ? 'more' : (after.modelIR.totalParams ?? 0) < before ? 'fewer' : 'same'
    setResults((current) => ({ ...current, [label]: outcome }))
  }
  const outcomeText: Record<Prediction, [string, string]> = { more: ['more parameters', '参数变多'], fewer: ['fewer parameters', '参数变少'], same: ['same parameter count', '参数不变'], error: ['an error', '报错'] }

  return <section className="inspector-section reading-guide">
    <h2><BookOpenCheck size={13} />{tr(locale, 'Read it in three passes', '三遍读懂这个模型')}<span className="reading-progress">{progress.map((ok, index) => <i key={index} className={ok ? 'ok' : ''} />)}</span></h2>
    <div className="pass-tabs" role="tablist">
      {([1, 2, 3] as const).map((value) => <button type="button" role="tab" key={value} aria-selected={pass === value} onClick={() => setPass(value)}>
        <b>{value}</b>{value === 1 ? tr(locale, 'Bird’s-eye', '鸟瞰') : value === 2 ? tr(locale, 'Structure', '抓结构') : tr(locale, 'Rebuild', '重现')}
      </button>)}
    </div>

    {pass === 1 && <div className="pass-body">
      <p className="pass-lead">{tr(locale, 'Five minutes: title, abstract, figures and conclusion. Skip the details.', '五分钟：只看标题、摘要、图和结论，先不管细节。')}</p>
      <div className="pass-facts">
        {facts.inputs.length > 0 && <div><span>{tr(locale, 'In', '输入')}</span>{facts.inputs.slice(0, 4).map((node) => <button type="button" key={node.id} onClick={() => selectNode(node.id)}><code>{node.variable ?? name(node)}</code>{node.outputShape?.join('×')}</button>)}</div>}
        {facts.outputs.length > 0 && <div><span>{tr(locale, 'Out', '输出')}</span>{facts.outputs.slice(0, 3).map((node) => <button type="button" key={node.id} onClick={() => selectNode(node.id)}><code>{name(node)}</code>{node.outputShape?.join('×')}</button>)}</div>}
        {model.totalParams !== undefined && <div><span>{tr(locale, 'Size', '规模')}</span><em>{formatCount(model.totalParams)} {tr(locale, 'params', '参数')} · {facts.blocks.length} {tr(locale, 'top-level blocks', '个顶层模块')}</em></div>}
      </div>
      <div className="pass-checks">{firstPassChecks.map((item) => <label key={item.key}><input type="checkbox" checked={done(item.key)} onChange={() => useStudyStore.getState().toggleCheck(key, item.key)} />{locale === 'zh' ? item.zh : item.en}</label>)}</div>
      <div className="pass-links">
        {reference && <a href={reference} target="_blank" rel="noreferrer"><ExternalLink size={12} />{tr(locale, 'Paper', '论文')}</a>}
        {origin === 'code' && study && <button type="button" onClick={() => useModelStore.getState().loadBuiltIn(study.model.id)}><MapIcon size={12} />{tr(locale, 'Paper figure', '论文结构图')}</button>}
      </div>
    </div>}

    {pass === 2 && <div className="pass-body">
      <p className="pass-lead">{tr(locale, 'Walk the main blocks in order; for each, find it in the code and say what changes in the tensor.', '按顺序走一遍主要模块：在代码里找到它，说出张量形状怎么变。')}</p>
      {facts.blocks.length ? <div className="pass-blocks">{facts.blocks.map((node, index) => {
        const share = node.paramCount && model.totalParams ? node.paramCount / model.totalParams : 0
        return <div key={node.id} className={`pass-block ${done(`p2:${node.id}`) ? 'done' : ''}`}>
          <button type="button" className="pass-check" onClick={() => useStudyStore.getState().toggleCheck(key, `p2:${node.id}`)} aria-label={tr(locale, 'Mark as understood', '标记为已理解')}>{done(`p2:${node.id}`) ? <Check size={11} /> : index + 1}</button>
          <button type="button" className="pass-block-main" onClick={() => selectNode(node.id)}>
            <span><i style={{ background: types[node.type] }} />{name(node)}<small>{node.className}</small></span>
            <em>{node.inputShape?.join('×') ?? '…'} → {node.outputShape?.join('×') ?? '…'}</em>
            {share > 0 && <b style={{ width: `${Math.max(2, share * 100)}%`, background: types[node.type] }} />}
          </button>
        </div>
      })}</div> : <p className="pass-empty">{tr(locale, 'This map has no code blocks — open the runnable version.', '这张图没有代码模块，可以打开可运行的代码版本。')}</p>}
      <p className="pass-tip"><Route size={12} />{tr(locale, 'Select one block, then Shift+click another to see how data gets there.', '选中一个模块，再 Shift+点击另一个，看数据怎样流过去。')}</p>
    </div>}

    {pass === 3 && <div className="pass-body">
      <p className="pass-lead">{tr(locale, 'Rebuild it in your head: predict what an edit will do, then run it. Change one thing at a time.', '在脑子里重新实现一遍：先预测改动的后果，再运行验证。一次只改一处。')}</p>
      {entry && entry.experiments.length > 0 && <div className="predict-list">
        {entry.experiments.map((experiment) => {
          const id = experiment.label.en
          const guess = answers?.[`predict:${id}`] as Prediction | undefined
          const result = results[id]
          const file = experiment.file ?? 'model.py'
          const available = useModelStore.getState().codeFiles[file]?.includes(experiment.find) ?? false
          return <div key={id} className="predict-item">
            <div className="predict-title"><FlaskConical size={12} />{experiment.label[locale]}</div>
            <div className="predict-options">{(['more', 'fewer', 'same', 'error'] as const).map((option) => <button type="button" key={option} aria-pressed={guess === option} onClick={() => useStudyStore.getState().setAnswer(key, `predict:${id}`, option)}>{tr(locale, ...outcomeText[option])}</button>)}</div>
            {result ? <div className={`predict-result ${guess === result ? 'right' : 'wrong'}`}>{guess === result ? <Check size={12} /> : <X size={12} />}{tr(locale, `Result: ${outcomeText[result][0]}`, `结果：${outcomeText[result][1]}`)}<small>{experiment.hint[locale]}</small></div>
              : <button type="button" className="predict-run" disabled={!guess || !available} onClick={() => runExperiment(id, file, experiment.find, experiment.replace)}>{available ? tr(locale, 'Apply and check', '应用并验证') : tr(locale, 'Already applied', '已应用')}</button>}
          </div>
        })}
        {diagnostic && <p className="pass-tip warn">{tr(locale, 'The code now fails — read the error, then Reset or fix it.', '代码现在报错：先读懂报错，再“重置”或动手修复。')}</p>}
      </div>}
      <div className="ng-questions">
        <div className="ng-title"><Lightbulb size={12} />{tr(locale, 'After reading (Andrew Ng’s four questions)', '读完回答（吴恩达四问）')}</div>
        {questions.map((question) => <label key={question.key}>
          <span>{locale === 'zh' ? question.zh : question.en}</span>
          <textarea rows={2} value={answers?.[question.key] ?? ''} placeholder={tr(locale, 'Your answer, saved in this browser', '你的回答，自动保存在本机浏览器')} onChange={(event) => useStudyStore.getState().setAnswer(key, question.key, event.target.value)} />
        </label>)}
      </div>
    </div>}
  </section>
}

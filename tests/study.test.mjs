import { test } from 'node:test'
import assert from 'node:assert/strict'
import { lineChanges } from '../src/core/study/lineDiff.ts'
import { decodeShare, encodeShare } from '../src/core/study/share.ts'
import { diffModels } from '../src/core/study/diff.ts'

test('line diff marks modified, added and deleted lines', () => {
  assert.deepEqual(lineChanges('a\nb\nc\nd', 'a\nB\nc\nx\nd'), { added: [4], modified: [2], deleted: [] })
  assert.deepEqual(lineChanges('a\nb\nc\nd', 'a\nd'), { added: [], modified: [], deleted: [1] })
  assert.deepEqual(lineChanges('same', 'same'), { added: [], modified: [], deleted: [] })
})

test('share links round-trip code, selection and notes', async () => {
  const state = { v: 1, model: 'code-gpt', files: { 'model.py': 'x = 1\n# 中文注释' }, selected: 'wte', view: '2d', depth: 2, notes: { wte: '词嵌入' }, path: 'lm_head' }
  const token = await encodeShare(state)
  assert.deepEqual(await decodeShare(token), state)
  assert.equal(await decodeShare('zbroken'), null)
})

test('model diff finds new blocks, reshapes and parameter changes', () => {
  const base = { id: 'm', name: 'm', totalParams: 10, nodes: [{ id: 'g', name: 'g', type: 'module' }, { id: 'a', name: 'a', type: 'linear', parentId: 'g', outputShape: [1, 4], paramCount: 10 }], edges: [] }
  const next = { id: 'm', name: 'm', totalParams: 30, nodes: [{ id: 'g', name: 'g', type: 'module' }, { id: 'a', name: 'a', type: 'linear', parentId: 'g', outputShape: [1, 8], paramCount: 20 }, { id: 'b', name: 'b', type: 'linear', outputShape: [1, 8], paramCount: 10 }], edges: [] }
  const diff = diffModels(base, next)
  assert.deepEqual(diff.added.map((node) => node.id), ['b'])
  assert.equal(diff.reshaped[0].node.id, 'a')
  assert.equal(diff.params[0].to, 20)
  assert.equal(diff.marks.get('g'), 'changed')
  assert.equal(diff.marks.get('b'), 'added')
})

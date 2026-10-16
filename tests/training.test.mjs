// Training-step tracing and the import template: run with `npm run test:training`.
import { test, after } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('..', import.meta.url))
const server = await createServer({ root, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error', optimizeDeps: { noDiscovery: true } })
after(() => server.close())
const { buildModelFromCode, executionOrder } = await server.ssrLoadModule('/src/core/pytorch/index.ts')
const template = Object.fromEntries(['model.py', 'data.py', 'train.py', 'atlas.json'].map((file) => [file, readFileSync(`${root}templates/atlas-template/${file}`, 'utf8')]))
const top = (model) => model.nodes.filter((node) => !node.parentId)

test('the template draws one training step: batch → model → loss → backward → clip → optimizer', () => {
  const { model } = buildModelFromCode('t', template)
  assert.deepEqual(top(model).map((node) => node.className), ['Input', 'Input', 'SmallCNN', 'CrossEntropyLoss', 'loss.backward', 'nn.utils.clip_grad_norm_', 'AdamW.step'])
  const [images, labels] = top(model)
  assert.equal(images.name, 'images')
  assert.deepEqual(images.outputShape, [64, 3, 32, 32])
  assert.equal(labels.name, 'labels')
  assert.deepEqual(labels.outputShape, [64])
  // Parameters match PyTorch: conv 864 + bn 64 + conv 18432 + bn 128 + conv 73728 + bn 256 + linear 1290.
  assert.equal(model.totalParams, 94762)
  const step = top(model).at(-1)
  assert.equal(step.type, 'optimizer')
  assert.equal(step.parameters.lr, 0.0003)
  assert.equal(step.parameters.scheduler, 'CosineAnnealingLR')
  assert.equal(step.parameters.loop, '20 × 781')
  assert.equal(step.parameters.steps, 15620)
  const edge = (a, b) => model.edges.some((item) => item.source === a && item.target === b)
  assert.ok(edge('labels', 'crossentropyloss') && edge('crossentropyloss', 'backward') && edge('backward', 'clip_grad_norm') && edge('clip_grad_norm', 'step'))
  assert.ok(model.edges.some((item) => item.source === 'head' && item.target === 'crossentropyloss'), 'logits flow from the model into the loss')
})

test('entry model.py draws only the network; train.py does not run', () => {
  const files = { ...template, 'atlas.json': JSON.stringify({ entry: 'model.py' }) }
  assert.deepEqual(executionOrder(files), ['data.py', 'model.py'])
  const { model } = buildModelFromCode('t', files)
  assert.ok(!model.nodes.some((node) => node.type === 'optimizer'))
  assert.deepEqual(top(model).at(-1).className, 'Output')
  assert.equal(model.totalParams, 94762)
})

test('files run after the local files they import, the training script last', () => {
  assert.deepEqual(executionOrder({ 'train.py': 'from model import Net\nfrom data import make\nloss.backward()', 'data.py': 'import torch', 'model.py': 'import torch' }), ['model.py', 'data.py', 'train.py'])
  assert.deepEqual(executionOrder({ 'model.py': 'import torch', 'utils.py': 'x = 1' }), ['utils.py', 'model.py'])
})

test('an example call in model.py is replaced by the training step; while loops, enumerate and GradScaler work', () => {
  const files = {
    'model.py': 'import torch\nimport torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(16, 4)\n    def forward(self, x):\n        return self.fc(x)\nmodel = Net()\ny = model(torch.randn(1, 16))\n',
    'train.py': [
      'import torch', 'import torch.nn.functional as F', 'from torch.utils.data import TensorDataset, DataLoader, random_split', 'from model import Net',
      'train, val = random_split(TensorDataset(torch.randn(1000, 16), torch.randint(0, 4, (1000,))), [0.8, 0.2])',
      'loader = DataLoader(train, batch_size=50)', 'model = Net()', 'opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)', 'scaler = torch.amp.GradScaler()', 'step = 0',
      'while step < 1000:', '    for i, (x, y) in enumerate(loader):', '        with torch.autocast("cuda"):', '            loss = F.cross_entropy(model(x), y)',
      '        opt.zero_grad()', '        scaler.scale(loss).backward()', '        scaler.step(opt)', '        scaler.update()', '        step += 1', '',
    ].join('\n'),
  }
  const { model } = buildModelFromCode('t', files)
  assert.deepEqual(top(model).map((node) => node.name), ['x', 'y', 'Net', 'cross_entropy', 'backward', 'SGD.step'])
  assert.deepEqual(top(model)[0].outputShape, [50, 16])
  assert.equal(top(model).at(-1).parameters.loop, '16')
  assert.equal(top(model).at(-1).parameters.momentum, 0.9)
  assert.equal(model.nodes.filter((node) => node.className === 'Linear').length, 1, 'the example forward pass is not drawn twice')
})

test('a user Dataset is sampled once without drawing its data preparation', () => {
  const files = {
    'model.py': 'import torch.nn as nn\nclass Net(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.emb = nn.Embedding(100, 8)\n    def forward(self, x):\n        return self.emb(x)\n',
    'train.py': [
      'import torch', 'from torch.utils.data import Dataset, DataLoader', 'from model import Net',
      'class Chunks(Dataset):', '    def __len__(self):', '        return 320', '    def __getitem__(self, i):', '        return {"ids": torch.randint(0, 100, (12,)), "target": torch.randn(12, 8)}',
      'model = Net()', 'opt = torch.optim.Adam(model.parameters())',
      'for epoch in range(2):', '    for batch in DataLoader(Chunks(), batch_size=32):', '        out = model(batch["ids"])', '        loss = ((out - batch["target"]) ** 2).mean()', '        loss.backward()', '        opt.step()', '',
    ].join('\n'),
  }
  const { model } = buildModelFromCode('t', files)
  const inputs = top(model).filter((node) => node.className === 'Input')
  assert.deepEqual(inputs.map((node) => [node.name, node.outputShape]), [['ids', [32, 12]], ['target', [32, 12, 8]]])
  assert.ok(!model.nodes.some((node) => node.className === 'torch.randint'), 'sampling inside __getitem__ is not part of the step')
  assert.equal(top(model).at(-1).parameters.loop, '2 × 10')
})

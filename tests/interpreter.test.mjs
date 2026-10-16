// The Python interpreter against real PyTorch: run with `npm run test:interpreter`.
// Each tests/interpreter/cases/<name>.py is a common PyTorch idiom; tests/interpreter/expected.json holds what
// PyTorch itself reports for it (parameter count and printed output), written by `npm run verify:torch -- --update`.
// To cover a new operation, add a case file and regenerate the expectations with PyTorch installed.
import { test, after } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { createServer } from 'vite'

const root = fileURLToPath(new URL('..', import.meta.url))
const server = await createServer({ root, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error', optimizeDeps: { noDiscovery: true } })
after(() => server.close())
const { buildModelFromCode } = await server.ssrLoadModule('/src/core/pytorch/index.ts')
const casesDir = `${root}tests/interpreter/cases`
const expected = JSON.parse(readFileSync(`${root}tests/interpreter/expected.json`, 'utf8'))
const build = (source) => buildModelFromCode('test', { 'model.py': source })
const header = 'import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n'

for (const file of readdirSync(casesDir).filter((name) => name.endsWith('.py')).sort()) {
  const name = file.slice(0, -3)
  test(`matches PyTorch: ${name}`, () => {
    assert.ok(expected[name], `no expectation for ${name}; run npm run verify:torch -- --update`)
    const { model, logs } = build(readFileSync(`${casesDir}/${file}`, 'utf8'))
    assert.equal(model.totalParams, expected[name].total, 'parameter count')
    assert.equal(logs.join('\n').trim(), expected[name].print, 'print() output')
  })
}

test('f-string names give each layer its own parameters', () => {
  const { model } = build(`${header}
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleDict({f"b{i}": nn.Linear(8, 8) for i in range(3)})
        for i in range(3):
            setattr(self, f"layer{i}", nn.Linear(8, 8))
    def forward(self, x):
        for i in range(3):
            x = getattr(self, f"layer{i}")(self.blocks[f"b{i}"](x))
        return x

if __name__ == "__main__":
    M()(torch.randn(2, 8))
`)
  assert.equal(model.totalParams, 6 * (8 * 8 + 8))
})

test('a file without a model is drawn as a script of tensor operations', () => {
  const { model, logs } = build(`${header}
x = torch.randn(2, 3)
y = (x @ x.T).softmax(-1)
print(y.shape)
`)
  assert.equal(model.totalParams, 0)
  assert.deepEqual(logs, ['torch.Size([2, 2])'])
  assert.ok(model.nodes.some((node) => node.className === 'matmul'))
})

test('a file with no tensor work explains what is missing', () => {
  assert.throws(() => build(`${header}
class M(nn.Module):
    def forward(self, x):
        return x
`), (error) => /No model was run/.test(error.diagnostic.message))
})

test('errors point at the line in the user file', () => {
  assert.throws(() => build(`${header}
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc(x)

M()(torch.randn(2, 8))
`), (error) => error.diagnostic.line === 8 && /in_features=16/.test(error.diagnostic.message))
  assert.throws(() => build('x = (1,\n'), (error) => /SyntaxError/.test(error.diagnostic.message))
})

test('the prelude classes report the user line, not their own', () => {
  const { model } = build(`${header}
layer = nn.TransformerEncoderLayer(32, 4, 64, batch_first=True)
enc = nn.TransformerEncoder(layer, num_layers=2)
enc(torch.randn(2, 5, 32))
`)
  for (const node of model.nodes) if (node.source) assert.equal(node.source.file, 'model.py')
})

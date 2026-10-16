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

// The checks below have no real-PyTorch case file; each expected value comes from the formula PyTorch documents.
const run = (source) => build(`${header}${source}\n_ = torch.randn(1)\n`).logs
const paramsOf = (layer, inputs) => build(`${header}layer = ${layer}\nlayer(${inputs})\n`).model.totalParams
const size = (...dims) => `torch.Size([${dims.join(', ')}])`
const fails = (source, pattern) => assert.throws(() => run(source), (error) => pattern.test(error.diagnostic.message))

test('pooling honours ceil_mode and dilation', () => {
  // out = floor_or_ceil((L + 2p - d*(k-1) - 1) / s) + 1; with ceil_mode, drop a last window that starts at or past L + p.
  assert.deepEqual(run(`
print(nn.MaxPool2d(2, ceil_mode=True)(torch.randn(1, 3, 7, 7)).shape)
print(nn.AvgPool2d(3, 2, ceil_mode=True)(torch.randn(1, 3, 8, 8)).shape)
print(nn.MaxPool2d(3, 1, dilation=2)(torch.randn(1, 3, 8, 8)).shape)
print(nn.MaxPool1d(3, 3, padding=1, ceil_mode=True)(torch.randn(1, 2, 5)).shape)
print(F.max_pool2d(torch.randn(1, 3, 7, 7), 2, ceil_mode=True).shape)
print(F.avg_pool2d(torch.randn(1, 3, 8, 8), 3, 2, 0, True).shape)
print(F.max_pool2d(torch.randn(1, 3, 8, 8), 3, 1, 0, 2).shape)
print(F.max_pool3d(torch.randn(1, 2, 7, 7, 7), 2, ceil_mode=True).shape)
`), [size(1, 3, 4, 4), size(1, 3, 4, 4), size(1, 3, 4, 4), size(1, 2, 2), size(1, 3, 4, 4), size(1, 3, 4, 4), size(1, 3, 4, 4), size(1, 2, 4, 4, 4)])
})

test('max pooling with return_indices returns (output, long indices)', () => {
  assert.deepEqual(run(`
out, idx = nn.MaxPool2d(2, return_indices=True)(torch.randn(1, 3, 8, 8))
print(out.shape, idx)
out, idx = F.max_pool1d(torch.randn(1, 3, 8), 2, return_indices=True)
print(idx)
out, idx = nn.AdaptiveMaxPool2d((None, 2), return_indices=True)(torch.randn(1, 3, 8, 6))
print(out.shape, idx)
`), [`${size(1, 3, 4, 4)} tensor(shape=[1, 3, 4, 4], dtype=torch.int64)`, 'tensor(shape=[1, 3, 4], dtype=torch.int64)', `${size(1, 3, 8, 2)} tensor(shape=[1, 3, 8, 2], dtype=torch.int64)`])
})

test('ConvTranspose weights are [in, out / groups, *kernel]', () => {
  // ConvTranspose2d(8, 8, 3, groups=4): 8 * 2 * 3 * 3 + 8 bias = 152
  assert.equal(paramsOf('nn.ConvTranspose2d(8, 8, 3, groups=4)', 'torch.randn(1, 8, 5, 5)'), 152)
  fails('nn.ConvTranspose2d(6, 8, 3, groups=4)', /groups=4 must divide/)
})

test('F.conv_transpose2d applies dilation', () => {
  // out = (L - 1) * s - 2p + d * (k - 1) + output_padding + 1 = 4 - 0 + 2 * 2 + 0 + 1 = 9; channels = weight.shape[1] * groups
  assert.deepEqual(run(`
x, w = torch.randn(1, 8, 5, 5), torch.randn(8, 4, 3, 3)
print(F.conv_transpose2d(x, w, None, 1, 0, 0, 1, 2).shape)
print(F.conv_transpose2d(x, w, dilation=2).shape)
`), [size(1, 4, 9, 9), size(1, 4, 9, 9)])
})

test('MultiheadAttention handles kdim/vdim, bias_kv, zero attention and weight options', () => {
  // kdim/vdim ≠ embed_dim: q/k/v_proj_weight 16*16 + 16*8 + 16*4 = 448, in_proj_bias 48, out_proj 16*16 + 16 = 272
  assert.equal(paramsOf('nn.MultiheadAttention(16, 4, kdim=8, vdim=4)', 'torch.randn(5, 2, 16), torch.randn(7, 2, 8), torch.randn(7, 2, 4)'), 768)
  // in_proj 3*16*16 + 48, out_proj 272, bias_k and bias_v [1, 1, 16]
  assert.equal(paramsOf('nn.MultiheadAttention(16, 4, add_bias_kv=True)', 'torch.randn(5, 2, 16), torch.randn(7, 2, 16), torch.randn(7, 2, 16)'), 1120)
  // bias_k/bias_v and add_zero_attn each add one key position to the [N, L, S] weights
  assert.deepEqual(run(`
q, k = torch.randn(5, 2, 16), torch.randn(7, 2, 16)
out, w = nn.MultiheadAttention(16, 4, kdim=8, vdim=4)(q, torch.randn(7, 2, 8), torch.randn(7, 2, 4))
print(out.shape, w.shape)
print(nn.MultiheadAttention(16, 4, add_bias_kv=True)(q, k, k)[1].shape)
print(nn.MultiheadAttention(16, 4, add_bias_kv=True, add_zero_attn=True)(q, k, k)[1].shape)
print(nn.MultiheadAttention(16, 4)(q, k, k, need_weights=False)[1])
print(nn.MultiheadAttention(16, 4)(q, k, k, average_attn_weights=False)[1].shape)
print(nn.MultiheadAttention(16, 4)(torch.randn(5, 16), torch.randn(7, 16), torch.randn(7, 16), average_attn_weights=False)[1].shape)
`), [`${size(5, 2, 16)} ${size(2, 5, 7)}`, size(2, 5, 8), size(2, 5, 9), 'None', size(2, 4, 5, 7), size(4, 5, 7)])
})

test('Tensor.mT transposes only the last two dimensions', () => {
  assert.deepEqual(run(`
x = torch.randn(2, 3, 4)
print(x.mT.shape, x.mH.shape, x.T.shape)
`), [`${size(2, 4, 3)} ${size(2, 4, 3)} ${size(4, 3, 2)}`])
  fails('torch.randn(3).mT', /at least 2 dimensions/)
})

test('torch.where(condition) returns one long index tensor per dimension', () => {
  assert.deepEqual(run(`
rows, cols = torch.where(torch.randn(3, 4) > 0)
print(rows, cols)
`), ['tensor(shape=[12], dtype=torch.int64) tensor(shape=[12], dtype=torch.int64)'])
})

test('a huge range is not built before the loop runs', () => {
  const started = Date.now()
  fails('for i in range(10**9):\n    pass', /Execution limit reached/)
  assert.ok(Date.now() - started < 10_000, 'the step limit stops the loop quickly')
  assert.deepEqual(run(`
for i in range(10**9):
    if i == 3:
        break
r = range(10**9)
print(i, len(r), r[5], r[-1], list(range(10, 0, -3)), len(range(5, 1)))
`), ['3 1000000000 5 999999999 [10, 7, 4, 1] 0'])
})

test('del removes names, items, attributes and module parameters', () => {
  const { model } = build(`${header}
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Parameter(torch.randn(3))
        self.seq = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        del self.b, self.c
        del self.seq[0]
    def forward(self, x):
        return self.seq[1](self.a(x))

M()(torch.randn(2, 4))
`)
  // a and the remaining seq Linear: 2 * (4 * 4 + 4); Sequential renumbers its children after a delete
  assert.equal(model.totalParams, 40)
  assert.deepEqual(run(`
d = {'a': 1, 'b': 2}
del d['a']
items = [0, 1, 2, 3, 4]
del items[1], items[0]
del items[::2]
x = 5
del x
try:
    print(x)
except NameError:
    print(d, items, 'x deleted')
`), ["{'b': 2} [3] x deleted"])
  fails('d = {}\ndel d["missing"]', /KeyError/)
})

test('global and nonlocal route assignments to the declaring scope', () => {
  assert.deepEqual(run(`
counter = 0
def bump():
    global counter
    counter += 1
bump(); bump()
def outer():
    total = 0
    def add(n):
        nonlocal total
        total += n
    add(2); add(3)
    return total
print(counter, outer())
`), ['2 5'])
  fails('def f():\n    nonlocal y\nf()', /no binding for nonlocal 'y'/)
})

test('while…else and try…else run their else blocks', () => {
  assert.deepEqual(run(`
n = 0
while n < 3:
    n += 1
else:
    n += 10
m = 0
while True:
    m += 1
    break
else:
    m = -1
try:
    ok = 1
except ValueError:
    ok = 2
else:
    ok = 3
try:
    raise ValueError('boom')
except ValueError:
    bad = 2
else:
    bad = 3
print(n, m, ok, bad)
`), ['13 1 3 2'])
})

test('argmax/argmin/any/all take keepdim positionally', () => {
  assert.deepEqual(run(`
x = torch.randn(4, 5)
print(x.argmax(1, True).shape, x.argmin(0, True).shape, (x > 0).any(1, True).shape, (x > 0).all(1, True).shape)
`), [`${size(4, 1)} ${size(1, 5)} ${size(4, 1)} ${size(4, 1)}`])
})

test('LSTM proj_size projects the hidden state', () => {
  // weight_ih 4H*in = 80*10, weight_hh 4H*proj = 80*5, biases 2*4H = 160, weight_hr proj*H = 5*20 → 1460
  assert.equal(paramsOf('nn.LSTM(10, 20, proj_size=5)', 'torch.randn(7, 3, 10)'), 1460)
  // a second layer reads proj-sized inputs: 80*5 + 80*5 + 160 + 100 = 1060
  assert.equal(paramsOf('nn.LSTM(10, 20, num_layers=2, proj_size=5)', 'torch.randn(7, 3, 10)'), 1460 + 1060)
  assert.deepEqual(run(`
out, (h, c) = nn.LSTM(10, 20, proj_size=5)(torch.randn(7, 3, 10))
print(out.shape, h.shape, c.shape)
`), [`${size(7, 3, 5)} ${size(1, 3, 5)} ${size(1, 3, 20)}`])
  fails('nn.GRU(10, 20, proj_size=5)', /only supported for nn.LSTM/)
})

test('adaptive pooling, Unflatten and squeeze follow PyTorch sizes', () => {
  assert.deepEqual(run(`
x = torch.randn(1, 3, 10, 12)
print(nn.AdaptiveAvgPool2d((None, 7))(x).shape, F.adaptive_avg_pool2d(x, (5, None)).shape)
print(nn.Unflatten(1, (2, -1))(torch.randn(3, 12)).shape)
y = torch.randn(2, 1, 3, 1)
print(y.squeeze((1, 3)).shape, y.squeeze((0, 1)).shape, torch.squeeze(y, (1, 3)).shape)
`), [`${size(1, 3, 10, 7)} ${size(1, 3, 5, 12)}`, size(3, 2, 6), `${size(2, 3)} ${size(2, 3, 1)} ${size(2, 3)}`])
})

test('dimension errors are raised where PyTorch raises them', () => {
  // stack and unsqueeze insert a dimension, so dim must lie in [-(ndim + 1), ndim]
  fails('torch.stack([torch.randn(2, 3), torch.randn(2, 3)], dim=3)', /dim=3 is out of range/)
  assert.deepEqual(run('print(torch.stack([torch.randn(2, 3), torch.randn(2, 3)], dim=-3).shape)'), [size(2, 2, 3)])
  fails('torch.randn(2, 3).permute(0, 0)', /repeated dims/)
  fails('torch.randn(2, 3).unsqueeze(3)', /dim=3 is out of range/)
  assert.deepEqual(run('print(torch.randn(2, 3).unsqueeze(-3).shape)'), [size(1, 2, 3)])
})

test('Python arithmetic and list semantics', () => {
  assert.deepEqual(run(`
print(divmod(-7, 2), divmod(7, -2), divmod(7, 2))
print(round(2.5), round(3.5), round(-2.5), round(0.5), round(2.6))
print(isinstance(3, float), isinstance(2.5, float), isinstance(3, int))
print(3 * [1], [1, 2] * 2, 2 * (0,), [1] * -1, 2 * 'ab')
xs = [3, 1, 2]
xs.sort()
ys = ['bb', 'a', 'ccc']
ys.sort(key=len, reverse=True)
xs.remove(2)
print(xs, ys, 0o17, 0O7)
`), ['(-4, 1) (-4, -1) (3, 1)', '2 4 -2 0 3', 'False True True', '[1, 1, 1] [1, 2, 1, 2] (0, 0) [] abab', "[1, 3] ['ccc', 'bb', 'a'] 15 7"])
  fails('[1].remove(5)', /x not in list/)
  // a RangeError that is not a stack overflow is a bad value, not deep recursion
  fails("s = 'ab' * 10**9", /^ValueError: Invalid string length/)
})

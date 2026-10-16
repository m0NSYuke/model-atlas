// Checks Atlas against real PyTorch (optional; needs Python with `pip install torch`).
//
//   npm run verify:torch              built-in models + interpreter cases: parameter counts, output shapes, print()
//   npm run verify:torch -- --update  also rewrite tests/interpreter/expected.json from PyTorch
//
// Set PYTHON=/path/to/python to choose the interpreter. Nothing is downloaded; without torch the check is skipped.
import { createServer } from 'vite'
import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, readdirSync, writeFileSync, mkdirSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const reference = join(root, 'scripts', 'torch_reference.py')
const casesDir = join(root, 'tests', 'interpreter', 'cases')
const expectedPath = join(root, 'tests', 'interpreter', 'expected.json')
const update = process.argv.includes('--update')

const python = [process.env.PYTHON, 'python3', 'python'].filter(Boolean).find((command) => spawnSync(command, ['-c', 'import torch'], { stdio: 'ignore' }).status === 0)
if (!python) {
  console.log('PyTorch was not found (tried $PYTHON, python3, python), so the comparison is skipped.\nInstall it with `pip install torch` to run this check.')
  process.exit(0)
}
const torch = (...args) => {
  const result = spawnSync(python, [reference, ...args], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  if (result.status !== 0) throw new Error(result.stderr || `torch_reference.py exited with ${result.status}`)
  return JSON.parse(result.stdout.trim().split('\n').pop())
}
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b)

const server = await createServer({ root, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error', optimizeDeps: { noDiscovery: true } })
let failures = 0
try {
  const { buildModelFromCode, executionOrder } = await server.ssrLoadModule('/src/core/pytorch/index.ts')
  const { codeModels } = await server.ssrLoadModule('/src/data/models/codeCatalog.ts')

  console.log('Built-in models (parameters, model output, direct children)')
  for (const entry of codeModels) {
    const folder = mkdtempSync(join(tmpdir(), 'atlas-'))
    for (const [file, source] of Object.entries(entry.files)) {
      mkdirSync(dirname(join(folder, file)), { recursive: true })
      writeFileSync(join(folder, file), source)
    }
    const order = executionOrder(entry.files)
    const expected = torch('project', folder, order[order.length - 1])
    let model
    try { model = buildModelFromCode(entry.id, entry.files).model } catch (error) {
      failures++
      console.log(`  ✗ ${entry.id}: Atlas failed: ${error.diagnostic?.message ?? error}`)
      continue
    }
    const problems = []
    if (expected.error) problems.push(`PyTorch raised ${expected.error}`)
    if (model.totalParams !== expected.total) problems.push(`parameters ${model.totalParams} ≠ ${expected.total}`)
    // Atlas names a module node after its attribute path; in a training step the model sits inside a "model" group.
    const byId = new Map(model.nodes.map((node) => [node.id, node]))
    let checked = 0
    for (const rootModule of expected.roots.slice(0, 1)) {
      for (const [name, shape] of Object.entries(rootModule.children)) {
        const node = byId.get(name) ?? byId.get(`model/${name}`)
        if (!node || !Array.isArray(shape) || Array.isArray(shape[0])) continue
        checked++
        if (!same(node.outputShape, shape)) problems.push(`${name} → ${JSON.stringify(node.outputShape)} ≠ ${JSON.stringify(shape)}`)
      }
    }
    failures += problems.length ? 1 : 0
    console.log(`  ${problems.length ? '✗' : '✓'} ${entry.id}: ${expected.total.toLocaleString('en-US')} parameters, ${checked} child shapes${problems.length ? `\n      ${problems.join('\n      ')}` : ''}`)
  }

  console.log('\nInterpreter cases (tests/interpreter/cases)')
  const expected = torch('cases', casesDir)
  for (const file of readdirSync(casesDir).filter((name) => name.endsWith('.py')).sort()) {
    const name = file.slice(0, -3)
    const want = expected[name]
    let got
    try {
      const { model, logs } = buildModelFromCode(name, { 'model.py': readFileSync(join(casesDir, file), 'utf8') })
      got = { total: model.totalParams, print: logs.join('\n').trim() }
    } catch (error) { got = { error: String(error.diagnostic?.message ?? error) } }
    const ok = !want.error && got.total === want.total && got.print === want.print
    failures += ok ? 0 : 1
    console.log(`  ${ok ? '✓' : '✗'} ${name}${ok ? '' : `\n      PyTorch: ${JSON.stringify(want)}\n      Atlas:   ${JSON.stringify(got)}`}`)
  }
  if (update) {
    writeFileSync(expectedPath, `${JSON.stringify(Object.fromEntries(Object.entries(expected).map(([name, value]) => [name, { total: value.total, print: value.print }])), null, 1)}\n`)
    console.log(`\nWrote ${expectedPath}`)
  }
} finally {
  await server.close()
}
console.log(failures ? `\n${failures} difference(s) from PyTorch.` : '\nEverything matches PyTorch.')
process.exitCode = failures ? 1 : 0

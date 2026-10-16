// Traces a project from the command line and prints its top-level graph (debugging aid).
//
//   node scripts/trace-file.mjs model.py [train.py …]      files in a folder
//   node scripts/trace-file.mjs project.json                {"model.py": "...", ...}
import { createServer } from 'vite'
import { readFileSync } from 'node:fs'
import { basename } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const paths = process.argv.slice(2).filter((arg) => !arg.startsWith('-'))
const files = paths.length === 1 && paths[0].endsWith('.json') && basename(paths[0]) !== 'atlas.json'
  ? JSON.parse(readFileSync(paths[0], 'utf8'))
  : Object.fromEntries(paths.map((path) => [basename(path), readFileSync(path, 'utf8')]))
const server = await createServer({ root, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error', optimizeDeps: { noDiscovery: true } })
try {
  const { buildModelFromCode, executionOrder } = await server.ssrLoadModule('/src/core/pytorch/index.ts')
  console.log('run order:', executionOrder(files).join(' → '))
  const { model, logs } = buildModelFromCode('cli', files)
  const top = new Set(model.nodes.filter((node) => !node.parentId).map((node) => node.id))
  for (const node of model.nodes) if (top.has(node.id) || process.argv.includes('-a')) {
    console.log(`${node.parentId ? '  ' : ''}${node.id.padEnd(22)} ${node.type.padEnd(10)} ${String(node.className).padEnd(16)} ${JSON.stringify(node.outputShape ?? [])} ${node.parameters ? JSON.stringify(node.parameters) : ''}`)
  }
  console.log('edges:', model.edges.filter((edge) => top.has(edge.source) && top.has(edge.target)).map((edge) => `${edge.source}→${edge.target}`).join('  '))
  console.log(`params: ${model.totalParams?.toLocaleString('en-US')} · nodes: ${model.nodes.length}`)
  if (logs.length) console.log('print():', logs.slice(0, 8).join(' | '))
} catch (error) {
  console.log('ERROR', error.diagnostic ?? error)
  process.exitCode = 1
} finally {
  await server.close()
}

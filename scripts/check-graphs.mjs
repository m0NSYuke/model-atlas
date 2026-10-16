// Structural and visual integrity check for every built-in model.
//
//   npm run check:graphs            summary per model
//   npm run check:graphs -- -v      also list every problem
//
// Checks (all must be zero to pass):
//   • ports      every edge connects two existing nodes, no self loops or duplicates
//   • shapes     a traced data edge carries a tensor whose shape the target actually received
//   • overlaps   no two blocks share space in the 3D layout or the flat diagram
//   • through    no line passes through a block it does not connect to
// Reported (kept as low as the layout allows):
//   • crossings  pairs of lines that cross in the front view
import { createServer } from 'vite'
import { fileURLToPath } from 'node:url'

const root = fileURLToPath(new URL('..', import.meta.url))
const verbose = process.argv.includes('-v')
const only = process.argv.slice(2).filter((arg) => !arg.startsWith('-'))
const server = await createServer({ root, server: { middlewareMode: true, hmr: false }, appType: 'custom', logLevel: 'error', optimizeDeps: { noDiscovery: true } })
const load = (path) => server.ssrLoadModule(path)
const { buildModelFromCode } = await load('/src/core/pytorch/index.ts')
const { buildViewGraph, autoDepth, depthCollapse, sceneMetrics } = await load('/src/core/layout/viewGraph.ts')
const { diagramMetrics } = await load('/src/components/scene/diagramGeometry.ts')
const { checkScene, checkDiagram } = await load('/src/components/scene/drawingCheck.ts')
const { codeModels } = await load('/src/data/models/codeCatalog.ts')
const { studyCatalog } = await load('/src/data/models/studyCatalog.ts')

let failed = 0
const rows = []
function report(id, kind, result) {
  const bad = result.ports.length + result.shapes.length + result.overlaps.length + result.through.length
  if (bad) failed++
  rows.push([id, kind, result.edges, result.ports.length, result.shapes.length, result.overlaps.length, result.through.length, result.crossings.length])
  if (verbose || bad) for (const key of ['ports', 'shapes', 'overlaps', 'through', ...(verbose ? ['crossings'] : [])]) for (const item of result[key]) console.log(`  ${id} ${key}: ${item}`)
}

for (const entry of codeModels) {
  if (only.length && !only.includes(entry.id)) continue
  const { model } = buildModelFromCode(entry.id, entry.files)
  const level = entry.depth ?? autoDepth(model)
  for (const [kind, metrics] of [['3d', sceneMetrics], ['2d', diagramMetrics(1.4)]]) {
    const graph = buildViewGraph(model, depthCollapse(model, level), metrics)
    report(entry.id, kind, kind === '3d' ? checkScene(model, graph) : checkDiagram(model, graph))
  }
}
for (const entry of studyCatalog) {
  if (only.length && !only.includes(entry.model.id)) continue
  const graph = buildViewGraph(entry.model, () => false)
  report(entry.model.id, '3d', checkScene(entry.model, graph))
  report(entry.model.id, '2d', checkDiagram(entry.model, graph))
}

console.log('\nmodel'.padEnd(26) + 'view  edges  ports  shapes  overlaps  through  crossings')
for (const [id, kind, edges, ...counts] of rows) console.log(`${id.padEnd(25)} ${kind.padEnd(5)} ${String(edges).padStart(5)}  ${counts.map((count, index) => String(count).padStart([5, 6, 8, 7, 9][index])).join('  ')}`)
console.log(failed ? `\n${failed} view(s) have broken ports, shapes, overlaps or lines through blocks.` : '\nAll ports, shapes and block spacing are valid.')
await server.close()
process.exit(failed ? 1 : 0)

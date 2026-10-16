// The 3D path of an edge. Lives in core so the layout can test what the scene will draw.
import { CubicBezierCurve3, CatmullRomCurve3, Vector3, type Curve } from 'three'
import type { ViewEdge, ViewNode } from './viewGraph'

export const STUB_LENGTH = 0.95

/** A smooth 3D path for an edge. Residual paths arc above the main lane, skip paths below. */
export function edgeCurve(edge: ViewEdge, a: ViewNode, b: ViewNode): Curve<Vector3> {
  const [ax, ay, az] = a.position
  const [bx, by, bz] = b.position
  if (edge.stub) {
    // A short tap rising into the target from below; its label names the source.
    const x = bx + (edge.stub.index - (edge.stub.count - 1) / 2) * 0.55
    const end = new Vector3(x, by - b.size[1] / 2, bz)
    const start = new Vector3(x, end.y - STUB_LENGTH, bz)
    return new CubicBezierCurve3(start, start.clone().lerp(end, 0.33), start.clone().lerp(end, 0.66), end)
  }
  if (edge.via?.length) {
    const first = edge.via[0]
    const last = edge.via[edge.via.length - 1]
    const exitX = Math.abs(first[0] - ax) > Math.abs(first[1] - ay)
    const enterX = Math.abs(last[0] - bx) > Math.abs(last[1] - by)
    const start = new Vector3(ax + (exitX ? Math.sign(first[0] - ax) * a.size[0] / 2 : 0), ay + (exitX ? 0 : Math.sign(first[1] - ay) * a.size[1] / 2), az)
    const end = new Vector3(bx + (enterX ? Math.sign(last[0] - bx) * b.size[0] / 2 : 0), by + (enterX ? 0 : Math.sign(last[1] - by) * b.size[1] / 2), bz)
    return new CatmullRomCurve3([start, ...edge.via.map((point) => new Vector3(...point)), end], false, 'centripetal', 0.2)
  }
  if (Math.abs(bx - ax) < (a.size[0] + b.size[0]) / 2 + 0.4 && Math.abs(by - ay) > (a.size[1] + b.size[1]) / 2) {
    // Stacked blocks (a step between resolution bands): leave and enter vertically.
    const down = by < ay ? -1 : 1
    const start = new Vector3(ax, ay + down * a.size[1] / 2, az)
    const end = new Vector3(bx, by - down * b.size[1] / 2, bz)
    const handle = Math.abs(end.y - start.y) * 0.45
    return new CubicBezierCurve3(start, new Vector3(start.x, start.y + down * handle, az), new Vector3(end.x, end.y - down * handle, bz), end)
  }
  const forward = bx >= ax
  const start = new Vector3(ax + (forward ? a.size[0] / 2 : -a.size[0] / 2), ay, az)
  const end = new Vector3(bx - (forward ? b.size[0] / 2 : -b.size[0] / 2), by, bz)
  const span = Math.abs(end.x - start.x)
  if (edge.type === 'residual' || edge.type === 'branch') {
    const above = edge.type === 'residual'
    const lift = (above ? 1 : -1) * (Math.max(a.size[1], b.size[1]) / 2 + 0.7 + Math.min(2.6, span * 0.08))
    const startTop = new Vector3(start.x - (forward ? a.size[0] / 2 : 0), ay + (above ? a.size[1] / 2 : -a.size[1] / 2), az)
    const endTop = new Vector3(end.x + (forward ? b.size[0] / 2 : 0), by + (above ? b.size[1] / 2 : -b.size[1] / 2), bz)
    const depth = above ? -0.5 : 0.5
    return new CubicBezierCurve3(startTop, new Vector3(startTop.x + span * 0.08, startTop.y + lift, az + depth), new Vector3(endTop.x - span * 0.08, endTop.y + lift, bz + depth), endTop)
  }
  const handle = Math.max(0.5, Math.abs(end.x - start.x) * 0.45)
  return new CubicBezierCurve3(start, new Vector3(start.x + (forward ? handle : -handle), start.y, start.z), new Vector3(end.x - (forward ? handle : -handle), end.y, end.z), end)
}

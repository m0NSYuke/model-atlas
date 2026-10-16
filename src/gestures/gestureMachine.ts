import type { HandIntent } from './handIntent'

export type GestureAction =
  | { type: 'select'; id: string }
  | { type: 'inspect'; id: string }
  | { type: 'rotate'; dx: number; dy: number }

interface Press {
  startX: number
  startY: number
  previousX: number
  previousY: number
  startedAt: number
  target: string | null
  moving: boolean
  inspected: boolean
}

export class GestureMachine {
  private press: Press | null = null

  reset() { this.press = null }

  update(sample: HandIntent, hoveredNodeId: string | null): GestureAction[] {
    if (!sample.pinched) {
      const ended = this.press
      this.press = null
      if (ended && !ended.moving && !ended.inspected && ended.target && sample.timestamp - ended.startedAt < 520) {
        return [{ type: 'select', id: ended.target }]
      }
      return []
    }

    if (!this.press) {
      this.press = { startX: sample.x, startY: sample.y, previousX: sample.x, previousY: sample.y, startedAt: sample.timestamp, target: hoveredNodeId, moving: false, inspected: false }
      return []
    }
    const press = this.press
    if (press.inspected) return []
    if (Math.hypot(sample.x - press.startX, sample.y - press.startY) > 0.035) press.moving = true
    const dx = Math.max(-0.055, Math.min(0.055, sample.x - press.previousX))
    const dy = Math.max(-0.055, Math.min(0.055, sample.y - press.previousY))
    press.previousX = sample.x
    press.previousY = sample.y
    if (press.moving) return [{ type: 'rotate', dx, dy }]
    if (press.target && sample.timestamp - press.startedAt >= 550) {
      press.inspected = true
      return [{ type: 'inspect', id: press.target }]
    }
    return []
  }
}

import type { HandIntent } from './handIntent'

export type GestureAction =
  | { type: 'select'; id: string }
  | { type: 'inspect'; id: string }
  | { type: 'rotate'; dx: number; dy: number }
  /** factor > 1 moves closer. The zoom centres on node `id` when set, otherwise on the screen point (x, y). */
  | { type: 'zoom'; factor: number; x: number; y: number; id: string | null }

interface Press {
  startX: number
  startY: number
  startedAt: number
  target: string | null
  moving: boolean
  inspected: boolean
}

interface Shape { timestamp: number; openness: number; x: number; y: number }

/** An open hand moves the model; below this it is relaxed, pointing or closing. */
export const OPEN = 0.72
/** Fingers folded in: a fist, a point or a pinch. */
export const CLOSED = 0.38
/** Opening or closing must happen within this time to count as a zoom gesture, so a slow curl does nothing. */
export const FLICK_MS = 450
export const ZOOM_STEP = 1.45
/** A node picked by a quick pinch stays the zoom target for this long. */
export const ANCHOR_MS = 4000
const OPEN_SETTLE_MS = 140
const AFTER_ZOOM_MS = 260

export class GestureMachine {
  private press: Press | null = null
  private shapes: Shape[] = []
  private openSince: number | null = null
  private previousPalm: { x: number; y: number } | null = null
  private lastZoomAt = -Infinity
  private anchor: { id: string; until: number } | null = null

  reset() {
    this.press = null
    this.shapes = []
    this.openSince = null
    this.previousPalm = null
    this.anchor = null
  }

  update(sample: HandIntent, hoveredNodeId: string | null): GestureAction[] {
    const now = sample.timestamp
    this.shapes.push({ timestamp: now, openness: sample.openness, x: sample.x, y: sample.y })
    while (this.shapes.length && now - this.shapes[0].timestamp > FLICK_MS) this.shapes.shift()
    if (this.anchor && now > this.anchor.until) this.anchor = null

    if (sample.pinched) {
      this.openSince = null
      this.previousPalm = null
      return this.holdPinch(sample, hoveredNodeId)
    }
    const actions = this.releasePinch(sample)
    const zoom = this.flick(sample)
    if (zoom) return [...actions, ...zoom]

    // Open palm: the model follows the hand. Pointing, pinching or a fist lets the hand move freely.
    if (sample.openness < OPEN) {
      this.openSince = null
      this.previousPalm = null
      return actions
    }
    if (this.openSince === null) this.openSince = now
    const previous = this.previousPalm
    this.previousPalm = { x: sample.palmX, y: sample.palmY }
    if (!previous || now - this.openSince < OPEN_SETTLE_MS || now - this.lastZoomAt < AFTER_ZOOM_MS) return actions
    const clamp = (value: number) => Math.max(-0.055, Math.min(0.055, value))
    const dx = clamp(sample.palmX - previous.x)
    const dy = clamp(sample.palmY - previous.y)
    if (Math.abs(dx) + Math.abs(dy) > 0.0015) actions.push({ type: 'rotate', dx, dy })
    return actions
  }

  /** A quick open is a zoom in at the point the hand opened over; a quick close into a fist zooms out. */
  private flick(sample: HandIntent): GestureAction[] | null {
    if (sample.timestamp - this.lastZoomAt < FLICK_MS) return null
    let opening = sample.openness >= OPEN
    let closing = sample.openness <= CLOSED && !sample.pointing
    if (!opening && !closing) return null
    // The pose the hand came from, within the flick window.
    const from = opening
      ? this.shapes.reduce((low, shape) => shape.openness < low.openness ? shape : low)
      : this.shapes.reduce((high, shape) => shape.openness > high.openness ? shape : high)
    // Right after a pinch picks a node, opening from a loose pinch is enough.
    opening &&= from.openness <= (this.anchor ? 0.5 : CLOSED)
    closing &&= from.openness >= OPEN
    if (!opening && !closing) return null
    this.lastZoomAt = sample.timestamp
    this.shapes = []
    this.previousPalm = null
    if (closing) return [{ type: 'zoom', factor: 1 / ZOOM_STEP, x: from.x, y: from.y, id: null }]
    const anchor = this.anchor
    this.anchor = null
    if (anchor) return [{ type: 'zoom', factor: ZOOM_STEP * 1.25, x: from.x, y: from.y, id: anchor.id }, { type: 'inspect', id: anchor.id }]
    return [{ type: 'zoom', factor: ZOOM_STEP, x: from.x, y: from.y, id: null }]
  }

  private holdPinch(sample: HandIntent, hoveredNodeId: string | null): GestureAction[] {
    if (!this.press) {
      this.press = { startX: sample.x, startY: sample.y, startedAt: sample.timestamp, target: hoveredNodeId, moving: false, inspected: false }
      return []
    }
    const press = this.press
    if (press.inspected || press.moving) return []
    // Pinching no longer drags; moving while pinched only cancels the click.
    if (Math.hypot(sample.x - press.startX, sample.y - press.startY) > 0.05) { press.moving = true; return [] }
    if (press.target && sample.timestamp - press.startedAt >= 650) {
      press.inspected = true
      return [{ type: 'inspect', id: press.target }]
    }
    return []
  }

  private releasePinch(sample: HandIntent): GestureAction[] {
    const ended = this.press
    this.press = null
    if (ended && !ended.moving && !ended.inspected && ended.target && sample.timestamp - ended.startedAt < 600) {
      this.anchor = { id: ended.target, until: sample.timestamp + ANCHOR_MS }
      return [{ type: 'select', id: ended.target }]
    }
    return []
  }
}

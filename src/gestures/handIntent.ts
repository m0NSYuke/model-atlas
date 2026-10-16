export interface Landmark { x: number; y: number; z: number }
export interface HandDetection { handedness: string; landmarks: Landmark[] }
export interface HandIntent {
  x: number
  y: number
  pinched: boolean
  pointing: boolean
  /** Palm centre (wrist and finger bases), which stays put while the fingers open or close. */
  palmX: number
  palmY: number
  /** 0 for a fist, 1 for a fully spread hand. */
  openness: number
  timestamp: number
}

const FINGERS = [[8, 5], [12, 9], [16, 13], [20, 17]] as const

/** How far the four fingers reach out from the wrist, relative to their own knuckles. */
export function handOpenness(points: Landmark[], aspect = 4 / 3): number {
  let sum = 0
  for (const [tip, base] of FINGERS) {
    const reach = distance(points[0], points[tip], aspect) / Math.max(distance(points[0], points[base], aspect), 0.01)
    // About 1.0 when the finger is curled into the palm, about 1.9 when straight.
    sum += Math.max(0, Math.min(1, (reach - 1.05) / 0.75))
  }
  return sum / FINGERS.length
}

function distance(a: Landmark, b: Landmark, aspect = 1) {
  return Math.hypot((a.x - b.x) * aspect, a.y - b.y)
}

class OneEuro {
  private value: number | null = null
  private derivative = 0
  private previousTime = 0

  reset() { this.value = null; this.derivative = 0; this.previousTime = 0 }

  update(next: number, time: number): number {
    if (this.value === null) { this.value = next; this.previousTime = time; return next }
    const dt = Math.max(0.001, Math.min(0.1, (time - this.previousTime) / 1000))
    this.previousTime = time
    const alpha = (cutoff: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt))
    const speed = (next - this.value) / dt
    this.derivative += alpha(1) * (speed - this.derivative)
    const cutoff = 1.1 + 0.42 * Math.abs(this.derivative)
    this.value += alpha(cutoff) * (next - this.value)
    return this.value
  }
}

export class HandIntentTracker {
  private x = new OneEuro()
  private y = new OneEuro()
  private palmX = new OneEuro()
  private palmY = new OneEuro()
  private open = new OneEuro()
  private pinched = false
  private pinchFrames = 0

  reset() { this.x.reset(); this.y.reset(); this.palmX.reset(); this.palmY.reset(); this.open.reset(); this.pinched = false; this.pinchFrames = 0 }

  update(points: Landmark[], timestamp: number, aspect = 4 / 3, mirror = true): HandIntent | null {
    if (points.length < 21) { this.reset(); return null }
    const palm = Math.max(distance(points[5], points[17], aspect), distance(points[0], points[9], aspect) * 0.85, 0.035)
    const pinchRatio = distance(points[4], points[8], aspect) / palm
    if (this.pinched) {
      if (pinchRatio > 0.57) { this.pinched = false; this.pinchFrames = 0 }
    } else {
      this.pinchFrames = pinchRatio < 0.37 ? this.pinchFrames + 1 : 0
      if (this.pinchFrames >= 2) this.pinched = true
    }
    const extended = distance(points[0], points[8], aspect) > distance(points[0], points[6], aspect) * 1.18
    const curled = ([12, 16, 20] as const).every((tip) => distance(points[0], points[tip], aspect) < distance(points[0], points[tip - 2], aspect) * 1.14)
    const pointing = extended && curled
    const rawX = mirror ? 1 - points[8].x : points[8].x
    const base = [0, 5, 9, 13, 17].map((index) => points[index])
    const rawPalmX = base.reduce((total, point) => total + point.x, 0) / base.length
    const rawPalmY = base.reduce((total, point) => total + point.y, 0) / base.length
    const clamp = (value: number) => Math.max(0, Math.min(1, value))
    return {
      x: Math.max(0, Math.min(1, this.x.update(rawX, timestamp))),
      y: Math.max(0, Math.min(1, this.y.update(points[8].y, timestamp))),
      pinched: this.pinched,
      pointing,
      palmX: clamp(this.palmX.update(mirror ? 1 - rawPalmX : rawPalmX, timestamp)),
      palmY: clamp(this.palmY.update(rawPalmY, timestamp)),
      openness: clamp(this.open.update(handOpenness(points, aspect), timestamp)),
      timestamp,
    }
  }
}

export interface Landmark { x: number; y: number; z: number }
export interface HandDetection { handedness: string; landmarks: Landmark[] }
export interface HandIntent {
  x: number
  y: number
  pinched: boolean
  pointing: boolean
  timestamp: number
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
  private pinched = false
  private pinchFrames = 0

  reset() { this.x.reset(); this.y.reset(); this.pinched = false; this.pinchFrames = 0 }

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
    return {
      x: Math.max(0, Math.min(1, this.x.update(rawX, timestamp))),
      y: Math.max(0, Math.min(1, this.y.update(points[8].y, timestamp))),
      pinched: this.pinched,
      pointing,
      timestamp,
    }
  }
}

import test from 'node:test'
import assert from 'node:assert/strict'
import { GestureMachine } from '../src/gestures/gestureMachine.ts'
import { HandIntentTracker, handOpenness } from '../src/gestures/handIntent.ts'

const OPEN_HAND = 0.95
const POINT = 0.25
const FIST = 0.05

const sample = (timestamp, pinched, x = 0.5, y = 0.5, openness = POINT) =>
  ({ timestamp, pinched, x, y, palmX: x, palmY: y + 0.15, openness, pointing: !pinched && openness === POINT })

/** Feeds samples every 40 ms (the tracker's frame rate) and collects every action. */
function run(machine, frames, hovered = 'block') {
  return frames.flatMap((frame) => machine.update(frame, hovered))
}

test('a short pinch selects the node captured at press time', () => {
  const machine = new GestureMachine()
  assert.deepEqual(machine.update(sample(0, true), 'attention'), [])
  assert.deepEqual(machine.update(sample(260, false), 'other'), [{ type: 'select', id: 'attention' }])
  assert.deepEqual(machine.update(sample(280, false), 'other'), [])
})

test('a steady pinch hold opens detail once and does not also select', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'transformer')
  assert.deepEqual(machine.update(sample(650, true), 'transformer'), [{ type: 'inspect', id: 'transformer' }])
  assert.deepEqual(machine.update(sample(680, true), 'transformer'), [])
  assert.deepEqual(machine.update(sample(700, false), 'transformer'), [])
})

test('moving while pinched no longer rotates, and cancels the click', () => {
  const machine = new GestureMachine()
  const actions = run(machine, [0, 40, 80, 120, 160].map((t, i) => sample(t, true, 0.5 + i * 0.03)))
  assert.deepEqual(actions, [])
  assert.deepEqual(machine.update(sample(200, false, 0.62), 'block'), [])
})

test('an open hand turns the model with the hand, without pinching', () => {
  const machine = new GestureMachine()
  // Hold an open hand still first so opening into the frame is not read as a zoom.
  const actions = run(machine, Array.from({ length: 12 }, (_, i) => sample(i * 40, false, 0.4 + Math.max(0, i - 5) * 0.02, 0.5, OPEN_HAND)))
  const rotations = actions.filter((action) => action.type === 'rotate')
  assert.ok(rotations.length >= 5, JSON.stringify(actions))
  assert.ok(rotations.every((action) => action.dx > 0 && action.dy === 0))
  assert.ok(actions.every((action) => action.type === 'rotate'))
})

test('moving the hand up gives a negative dy (the scene tips the model up)', () => {
  const machine = new GestureMachine()
  const actions = run(machine, Array.from({ length: 12 }, (_, i) => sample(i * 40, false, 0.5, 0.6 - Math.max(0, i - 5) * 0.02, OPEN_HAND)))
  const rotations = actions.filter((action) => action.type === 'rotate')
  assert.ok(rotations.length > 0 && rotations.every((action) => action.dy < 0))
})

test('pointing or a fist moves the cursor without turning the model', () => {
  const machine = new GestureMachine()
  assert.deepEqual(run(machine, Array.from({ length: 10 }, (_, i) => sample(i * 40, false, 0.3 + i * 0.03, 0.5, POINT))), [])
  assert.deepEqual(run(machine, Array.from({ length: 10 }, (_, i) => sample(1000 + i * 40, false, 0.3 + i * 0.03, 0.5, FIST))), [])
})

test('a quick fist-to-open zooms in at the point the hand opened over', () => {
  const machine = new GestureMachine()
  const frames = [FIST, FIST, 0.3, 0.6, 0.85, OPEN_HAND, OPEN_HAND].map((openness, i) => sample(i * 40, false, 0.3, 0.7, openness))
  const zooms = run(machine, frames).filter((action) => action.type === 'zoom')
  assert.equal(zooms.length, 1)
  assert.ok(zooms[0].factor > 1)
  assert.deepEqual([zooms[0].x, zooms[0].y, zooms[0].id], [0.3, 0.7, null])
})

test('a quick close into a fist zooms out; a slow curl does nothing', () => {
  const quick = new GestureMachine()
  const quickActions = run(quick, [OPEN_HAND, OPEN_HAND, 0.6, 0.3, FIST].map((openness, i) => sample(i * 40, false, 0.5, 0.5, openness)))
  const zooms = quickActions.filter((action) => action.type === 'zoom')
  assert.equal(zooms.length, 1)
  assert.ok(zooms[0].factor < 1)

  const slow = new GestureMachine()
  const slowActions = run(slow, Array.from({ length: 30 }, (_, i) => sample(i * 40, false, 0.5, 0.5, OPEN_HAND - (OPEN_HAND - FIST) * i / 29)))
  assert.deepEqual(slowActions.filter((action) => action.type === 'zoom'), [])
})

test('zooming in again: open, point (no zoom out), then open again', () => {
  const machine = new GestureMachine()
  const frames = [FIST, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND, OPEN_HAND,
    0.5, POINT, POINT, POINT, POINT, POINT, POINT, POINT, POINT, POINT, POINT, POINT, POINT, OPEN_HAND]
  const zooms = run(machine, frames.map((openness, i) => sample(i * 40, false, 0.5, 0.5, openness))).filter((action) => action.type === 'zoom')
  assert.equal(zooms.length, 2)
  assert.ok(zooms.every((action) => action.factor > 1))
})

test('pinch a node, then open the hand: zoom into that node and show its detail', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true, 0.6, 0.4), 'attention')
  assert.deepEqual(machine.update(sample(200, false, 0.6, 0.4, 0.45), null), [{ type: 'select', id: 'attention' }])
  const actions = run(machine, [0.45, 0.45, 0.7, OPEN_HAND].map((openness, i) => sample(240 + i * 40, false, 0.2, 0.8, openness)), null)
  const zoom = actions.find((action) => action.type === 'zoom')
  assert.equal(zoom?.id, 'attention')
  assert.ok(zoom.factor > 1)
  assert.deepEqual(actions.find((action) => action.type === 'inspect'), { type: 'inspect', id: 'attention' })
})

test('the pinched node stops being the zoom target after a few seconds', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'attention')
  machine.update(sample(200, false, 0.5, 0.5, FIST), null)
  const actions = run(machine, [FIST, OPEN_HAND].map((openness, i) => sample(5000 + i * 40, false, 0.5, 0.5, openness)), null)
  assert.deepEqual(actions.map((action) => [action.type, action.id]), [['zoom', null]])
})

test('tracking loss cancels a pending pinch', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'block')
  machine.reset()
  assert.deepEqual(machine.update(sample(100, false), 'block'), [])
})

/** A right hand seen by the camera: wrist at the bottom, fingers pointing up, each finger straight or curled. */
function hand({ index = true, middle = true, ring = true, pinky = true, pinch = false } = {}) {
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }))
  points[0] = { x: 0.5, y: 0.9, z: 0 }
  const fingers = [[5, index, 0.44], [9, middle, 0.5], [13, ring, 0.56], [17, pinky, 0.62]]
  for (const [base, straight, x] of fingers) {
    points[base] = { x, y: 0.66, z: 0 }
    points[base + 1] = { x, y: straight ? 0.56 : 0.6, z: 0 }
    points[base + 2] = { x, y: straight ? 0.49 : 0.68, z: 0 }
    points[base + 3] = { x, y: straight ? 0.43 : 0.74, z: 0 }
  }
  points[4] = pinch ? { ...points[8], x: points[8].x - 0.005 } : { x: 0.32, y: 0.62, z: 0 }
  return points
}

test('openness separates an open hand, a point and a fist', () => {
  assert.ok(handOpenness(hand()) > 0.85)
  assert.ok(handOpenness(hand({ middle: false, ring: false, pinky: false })) < 0.38)
  assert.ok(handOpenness(hand({ index: false, middle: false, ring: false, pinky: false })) < 0.1)
})

test('real landmarks: fist then open hand, through the tracker, zooms in', () => {
  const tracker = new HandIntentTracker()
  const machine = new GestureMachine()
  const poses = [...Array(6).fill(hand({ index: false, middle: false, ring: false, pinky: false })), ...Array(5).fill(hand())]
  const actions = poses.flatMap((points, i) => machine.update(tracker.update(points, i * 40), null))
  assert.equal(actions.filter((action) => action.type === 'zoom' && action.factor > 1).length, 1)
})

test('real landmarks: an open hand sliding to the right turns the model', () => {
  const tracker = new HandIntentTracker()
  const machine = new GestureMachine()
  const actions = []
  for (let i = 0; i < 16; i += 1) {
    // Mirror control is on, so moving left in camera space is moving right on screen.
    const shift = -Math.max(0, i - 5) * 0.015
    const points = hand().map((point) => ({ ...point, x: point.x + shift }))
    actions.push(...machine.update(tracker.update(points, i * 40), null))
  }
  const rotations = actions.filter((action) => action.type === 'rotate')
  assert.ok(rotations.length >= 5 && rotations.every((action) => action.dx > 0))
  assert.ok(!actions.some((action) => action.type === 'zoom'))
})

test('pinch hysteresis, pointing, and mirror mapping use normalized hand landmarks', () => {
  const tracker = new HandIntentTracker()
  const points = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.7, z: 0 }))
  points[0] = { x: 0.5, y: 0.9, z: 0 }
  points[5] = { x: 0.47, y: 0.62, z: 0 }
  points[6] = { x: 0.45, y: 0.45, z: 0 }
  points[8] = { x: 0.43, y: 0.2, z: 0 }
  points[17] = { x: 0.75, y: 0.65, z: 0 }
  for (const [tip, pip, x] of [[12, 10, 0.56], [16, 14, 0.62], [20, 18, 0.74]]) {
    points[tip] = { x, y: 0.76, z: 0 }
    points[pip] = { x, y: 0.55, z: 0 }
  }
  points[4] = { x: 0.72, y: 0.42, z: 0 }
  const pointing = tracker.update(points, 0)
  assert.equal(pointing.pointing, true)
  assert.equal(pointing.pinched, false)
  assert.ok(Math.abs(pointing.x - 0.57) < 0.001)
  points[4] = { x: 0.435, y: 0.22, z: 0 }
  assert.equal(tracker.update(points, 40).pinched, false)
  assert.equal(tracker.update(points, 80).pinched, true)
  points[4] = { x: 0.48, y: 0.23, z: 0 }
  assert.equal(tracker.update(points, 120).pinched, true)
  points[4] = { x: 0.72, y: 0.42, z: 0 }
  assert.equal(tracker.update(points, 160).pinched, false)
})

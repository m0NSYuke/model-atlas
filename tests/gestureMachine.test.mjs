import test from 'node:test'
import assert from 'node:assert/strict'
import { GestureMachine } from '../src/gestures/gestureMachine.ts'
import { HandIntentTracker } from '../src/gestures/handIntent.ts'

const sample = (timestamp, pinched, x = 0.5, y = 0.5) => ({ timestamp, pinched, x, y, pointing: !pinched })

test('a short pinch selects the node captured at press time', () => {
  const machine = new GestureMachine()
  assert.deepEqual(machine.update(sample(0, true), 'attention'), [])
  assert.deepEqual(machine.update(sample(260, false), 'other'), [{ type: 'select', id: 'attention' }])
  assert.deepEqual(machine.update(sample(280, false), 'other'), [])
})

test('a steady hold opens detail once and does not also select', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'transformer')
  assert.deepEqual(machine.update(sample(550, true), 'transformer'), [{ type: 'inspect', id: 'transformer' }])
  assert.deepEqual(machine.update(sample(580, true), 'transformer'), [])
  assert.deepEqual(machine.update(sample(600, false), 'transformer'), [])
})

test('movement rotates and cancels both selection and long hold', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'block')
  assert.deepEqual(machine.update(sample(80, true, 0.54), 'block'), [{ type: 'rotate', dx: 0.040000000000000036, dy: 0 }])
  assert.deepEqual(machine.update(sample(700, false, 0.54), 'block'), [])
})

test('tracking loss cancels a pending pinch', () => {
  const machine = new GestureMachine()
  machine.update(sample(0, true), 'block')
  machine.reset()
  assert.deepEqual(machine.update(sample(100, false), 'block'), [])
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

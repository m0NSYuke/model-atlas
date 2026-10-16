import { useRef, type RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Plane, Raycaster, Vector2, Vector3, type Object3D } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useModelStore } from '../../store/modelStore'
import { useUiStore } from '../../store/uiStore'
import type { HandIntent } from '../../gestures/handIntent'
import { GestureMachine } from '../../gestures/gestureMachine'

function nodeIdFor(object: Object3D | null): string | null {
  for (let current = object; current; current = current.parent) {
    if (typeof current.userData.nodeId === 'string') return current.userData.nodeId as string
  }
  return null
}

export function GestureSceneInput({ intentRef, controls, sensitivity, mousePauseUntil, onInspect }: {
  intentRef: RefObject<HandIntent | null>
  controls: RefObject<OrbitControlsImpl>
  sensitivity: number
  mousePauseUntil: RefObject<number>
  onInspect: (id: string) => void
}) {
  const { scene, camera } = useThree()
  const raycaster = useRef(new Raycaster())
  const pointer = useRef(new Vector2())
  const lastTimestamp = useRef(-1)
  const hovered = useRef<string | null>(null)
  const machine = useRef(new GestureMachine())
  /** Remaining zoom, as a log factor, eased in over the next frames around `point`. */
  const zoom = useRef<{ point: Vector3; remaining: number } | null>(null)

  const castAt = (x: number, y: number) => {
    pointer.current.set(x * 2 - 1, 1 - y * 2)
    raycaster.current.setFromCamera(pointer.current, camera)
    return raycaster.current.intersectObjects(scene.children, true).find((item) => nodeIdFor(item.object) !== null) ?? null
  }

  /** The world point a zoom should close in on: the node itself, what the hand is over, or the orbit plane behind it. */
  const zoomPoint = (x: number, y: number, id: string | null, control: OrbitControlsImpl) => {
    if (id) {
      let found: Object3D | null = null
      scene.traverse((object) => { if (!found && object.userData.nodeId === id) found = object })
      if (found) return (found as Object3D).getWorldPosition(new Vector3())
    }
    const hit = castAt(x, y)
    if (hit) return hit.point.clone()
    const normal = camera.getWorldDirection(new Vector3())
    return raycaster.current.ray.intersectPlane(new Plane().setFromNormalAndCoplanarPoint(normal, control.target), new Vector3()) ?? control.target.clone()
  }

  useFrame((_, delta) => {
    const control = controls.current
    const pending = zoom.current
    if (pending && control) {
      const step = pending.remaining * (1 - Math.exp(-Math.min(delta, 0.25) * 8))
      const factor = Math.exp(step)
      const distance = camera.position.distanceTo(control.target) / factor
      if (distance < control.minDistance || distance > control.maxDistance || Math.abs(pending.remaining) < 0.002) zoom.current = null
      else {
        // Scale the camera and orbit centre towards the point, so it stays under the hand while the view closes in.
        camera.position.sub(pending.point).divideScalar(factor).add(pending.point)
        control.target.sub(pending.point).divideScalar(factor).add(pending.point)
        control.update()
        pending.remaining -= step
      }
    }

    const sample = intentRef.current
    if (!sample) {
      machine.current.reset()
      lastTimestamp.current = -1
      if (hovered.current) { hovered.current = null; useModelStore.getState().hoverNode(null) }
      return
    }
    if (sample.timestamp === lastTimestamp.current || performance.now() < (mousePauseUntil.current ?? 0)) return
    lastTimestamp.current = sample.timestamp

    if (!sample.pinched) {
      const hit = sample.pointing || sample.openness < 0.72 ? nodeIdFor(castAt(sample.x, sample.y)?.object ?? null) : null
      if (hit !== hovered.current) { hovered.current = hit; useModelStore.getState().hoverNode(hit) }
    }
    for (const action of machine.current.update(sample, hovered.current)) {
      if (action.type === 'select') useModelStore.getState().selectNode(action.id)
      else if (action.type === 'inspect') {
        // Already selected (a pinch then an open hand): show its detail without gliding the camera away from the zoom.
        if (useModelStore.getState().selectedNodeId === action.id) useUiStore.getState().setTrace(true)
        else onInspect(action.id)
      } else if (!control) continue
      else if (action.type === 'zoom') {
        zoom.current = { point: zoomPoint(action.x, action.y, action.id, control), remaining: Math.log(action.factor) }
      } else {
        // The model follows the hand: right turns its front to the right, up tips its front upwards.
        control.setAzimuthalAngle(control.getAzimuthalAngle() - action.dx * 5.2 * sensitivity)
        control.setPolarAngle(Math.max(control.minPolarAngle + 0.01, Math.min(control.maxPolarAngle - 0.01, control.getPolarAngle() - action.dy * 4.2 * sensitivity)))
        control.update()
      }
    }
  })

  return null
}

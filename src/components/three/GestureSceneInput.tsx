import { useRef, type RefObject } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { Raycaster, Vector2, type Object3D } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { useModelStore } from '../../store/modelStore'
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

  useFrame(() => {
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
      pointer.current.set(sample.x * 2 - 1, 1 - sample.y * 2)
      raycaster.current.setFromCamera(pointer.current, camera)
      const hit = sample.pointing
        ? raycaster.current.intersectObjects(scene.children, true).map((item) => nodeIdFor(item.object)).find((id) => id !== null) ?? null
        : null
      if (hit !== hovered.current) { hovered.current = hit; useModelStore.getState().hoverNode(hit) }
    }
    for (const action of machine.current.update(sample, hovered.current)) {
      if (action.type === 'select') useModelStore.getState().selectNode(action.id)
      else if (action.type === 'inspect') onInspect(action.id)
      else {
        const control = controls.current
        if (control) {
          control.setAzimuthalAngle(control.getAzimuthalAngle() - action.dx * 5.2 * sensitivity)
          control.setPolarAngle(Math.max(control.minPolarAngle + 0.01, Math.min(control.maxPolarAngle - 0.01, control.getPolarAngle() + action.dy * 4.2 * sensitivity)))
          control.update()
        }
      }
    }
  })

  return null
}

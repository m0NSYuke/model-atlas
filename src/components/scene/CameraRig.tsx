import { useEffect, useRef, type RefObject } from 'react'
import { OrbitControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { PerspectiveCamera, Vector3 } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import type { ViewGraph } from '../../core/layout/viewGraph'
import { useModelStore } from '../../store/modelStore'

export type CameraAngle = 'iso' | 'front'

/** Fits the whole graph, glides to the selected node, and keeps orbit controls in sync. */
export function CameraRig({ graph, modelId, fitRequest, angle, controls }: { graph: ViewGraph; modelId: string; fitRequest: number; angle: CameraAngle; controls: RefObject<OrbitControlsImpl> }) {
  const { camera, size } = useThree()
  const selectedNodeId = useModelStore((state) => state.selectedNodeId)
  const focusRevision = useModelStore((state) => state.focusRevision)
  const handledFocus = useRef(focusRevision)
  const target = useRef<Vector3 | null>(null)
  const position = useRef<Vector3 | null>(null)
  const firstFit = useRef(true)

  const fit = (animate: boolean) => {
    const { min, max } = graph.bounds
    const center = new Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2)
    const aspect = size.width / Math.max(1, size.height)
    const fov = camera instanceof PerspectiveCamera ? camera.fov : 40
    const tan = Math.tan((fov * Math.PI) / 360)
    const direction = angle === 'iso' ? new Vector3(-0.3, 0.34, 1).normalize() : new Vector3(0, 0, 1)
    // Camera basis for this viewing direction; solve the distance that keeps every bounding-box corner in view.
    const right = new Vector3().crossVectors(new Vector3(0, 1, 0), direction).normalize()
    const up = new Vector3().crossVectors(direction, right).normalize()
    // The model card and view controls float over the top of the stage, the legend strip over the bottom,
    // the tool column over the right edge: fit into the clear area between them.
    const [top, bottom, side] = [140, 78, 70]
    const clearHeight = Math.max(0.45, (size.height - top - bottom) / Math.max(1, size.height))
    const clearWidth = Math.max(0.6, (size.width - side) / Math.max(1, size.width))
    let distance = 8
    for (const x of [min[0] - 1.2, max[0] + 1.2]) for (const y of [min[1] - 1.4, max[1] + 1.6]) for (const z of [min[2], max[2]]) {
      const corner = new Vector3(x, y, z).sub(center)
      const depth = corner.dot(direction)
      distance = Math.max(distance, depth + Math.abs(corner.dot(right)) / (tan * aspect * clearWidth), depth + Math.abs(corner.dot(up)) / (tan * clearHeight))
    }
    distance = Math.min(220, distance * 1.04)
    // Center the model in the clear area rather than in the whole canvas.
    center.addScaledVector(up, ((top - bottom) / Math.max(1, size.height)) * distance * tan)
    center.addScaledVector(right, (side / 2 / Math.max(1, size.width)) * 2 * distance * tan * aspect)
    const destination = center.clone().addScaledVector(direction, distance)
    if (animate) { target.current = center; position.current = destination; return }
    camera.position.copy(destination)
    camera.lookAt(center)
    controls.current?.target.copy(center)
    controls.current?.update()
    target.current = null
    position.current = null
  }

  // Refit when the model changes or the user asks; keep the camera when the same model is re-traced.
  // modelId arrives as a prop together with graph: reading it from the store here could run ahead of the new graph.
  useEffect(() => { fit(!firstFit.current); firstFit.current = false }, [fitRequest, modelId, angle]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (focusRevision === handledFocus.current) return
    handledFocus.current = focusRevision
    if (!selectedNodeId) return
    const id = graph.representative.get(selectedNodeId) ?? selectedNodeId
    const item = graph.byId.get(id)
    const frame = graph.frames.find((entry) => entry.id === selectedNodeId)
    const point = item ? new Vector3(...item.position) : frame ? new Vector3((frame.min[0] + frame.max[0]) / 2, (frame.min[1] + frame.max[1]) / 2, 0) : null
    const control = controls.current
    if (!point || !control) return
    const offset = camera.position.clone().sub(control.target)
    // Move closer when the whole model is in view, so the selected layer becomes readable.
    if (offset.length() > 26) offset.setLength(Math.max(18, offset.length() * 0.6))
    target.current = point
    position.current = point.clone().add(offset)
  }, [focusRevision, selectedNodeId, graph, camera, controls])

  useFrame((_, delta) => {
    const control = controls.current
    if (!control || !target.current || !position.current) return
    // Frame-rate independent easing, so slow GPUs still finish the move quickly.
    const alpha = 1 - Math.exp(-Math.min(delta, 0.25) * 7)
    control.target.lerp(target.current, alpha)
    camera.position.lerp(position.current, alpha)
    control.update()
    if (control.target.distanceTo(target.current) < 0.01 && camera.position.distanceTo(position.current) < 0.02) { target.current = null; position.current = null }
  })

  return <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.09} minDistance={3} maxDistance={180} maxPolarAngle={Math.PI * 0.86}
    onStart={() => { target.current = null; position.current = null }} screenSpacePanning />
}

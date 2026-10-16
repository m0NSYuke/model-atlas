import { useCallback, useEffect, useRef, useState } from 'react'
import { HandIntentTracker, type HandDetection, type HandIntent } from './handIntent'

export type TrackingStatus = 'off' | 'starting' | 'loading' | 'searching' | 'tracking' | 'error'
export type HandPreference = 'any' | 'left' | 'right'
export type TrackingError = 'permission' | 'notFound' | 'inUse' | 'unsupported' | 'timeout' | 'model' | 'unknown' | null

interface WorkerMessage {
  type: 'ready' | 'result' | 'error'
  timestamp?: number
  hands?: HandDetection[]
  message?: string
}

function errorKind(error: unknown): TrackingError {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') return 'permission'
    if (error.name === 'NotFoundError' || error.name === 'OverconstrainedError') return 'notFound'
    if (error.name === 'NotReadableError') return 'inUse'
  }
  return 'unknown'
}

export function useHandTracking() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const intentRef = useRef<HandIntent | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const workerRef = useRef<Worker | null>(null)
  const frameRef = useRef<number | null>(null)
  const sessionRef = useRef(0)
  const busyRef = useRef(false)
  const lastFrameRef = useRef(0)
  const trackerRef = useRef(new HandIntentTracker())
  const preferredRef = useRef<HandPreference>('any')
  const mirrorRef = useRef(true)
  const [status, setStatus] = useState<TrackingStatus>('off')
  const [error, setError] = useState<TrackingError>(null)
  const [errorDetail, setErrorDetail] = useState<string | null>(null)
  const [uiIntent, setUiIntent] = useState<HandIntent | null>(null)
  const [preferredHand, setPreferredHand] = useState<HandPreference>('any')
  const [mirror, setMirror] = useState(true)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [sensitivity, setSensitivity] = useState(1)

  useEffect(() => { preferredRef.current = preferredHand; trackerRef.current.reset() }, [preferredHand])
  useEffect(() => { mirrorRef.current = mirror; trackerRef.current.reset() }, [mirror])

  const stop = useCallback(() => {
    sessionRef.current += 1
    if (frameRef.current !== null) cancelAnimationFrame(frameRef.current)
    frameRef.current = null
    workerRef.current?.terminate()
    workerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    busyRef.current = false
    trackerRef.current.reset()
    intentRef.current = null
    setUiIntent(null)
    setStatus('off')
    setError(null)
    setErrorDetail(null)
  }, [])

  const start = useCallback(async (requestedDeviceId = deviceId) => {
    stop()
    const session = sessionRef.current
    if (!navigator.mediaDevices?.getUserMedia || typeof Worker === 'undefined' || typeof createImageBitmap === 'undefined') {
      setStatus('error'); setError('unsupported'); return
    }
    setStatus('starting')
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    try {
      let timedOut = false
      const cameraRequest = navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { ...(requestedDeviceId ? { deviceId: { exact: requestedDeviceId } } : { facingMode: 'user' }), width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } },
      })
      // A browser can leave a permission request pending indefinitely. Release a late stream.
      cameraRequest.then((lateStream) => {
        if (timedOut || session !== sessionRef.current) lateStream.getTracks().forEach((track) => track.stop())
      }).catch(() => {})
      const stream = await Promise.race([
        cameraRequest,
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => { timedOut = true; reject(new DOMException('Camera request timed out', 'TimeoutError')) }, 12000)
        }),
      ])
      if (timeoutId) clearTimeout(timeoutId)
      if (session !== sessionRef.current) { stream.getTracks().forEach((track) => track.stop()); return }
      streamRef.current = stream
      const video = videoRef.current
      if (!video) throw new Error('Video preview is unavailable')
      video.srcObject = stream
      await video.play()
      if (session !== sessionRef.current) return
      setDevices((await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'videoinput'))
      setDeviceId(stream.getVideoTracks()[0]?.getSettings().deviceId ?? requestedDeviceId)
      setStatus('loading')
      const assetBase = new URL(`${import.meta.env.BASE_URL}mediapipe`, window.location.origin).href.replace(/\/$/, '')
      const worker = new Worker(`${assetBase}/hand-worker.js`)
      workerRef.current = worker
      const capture = async (now: number) => {
        if (session !== sessionRef.current) return
        frameRef.current = requestAnimationFrame(capture)
        if (busyRef.current || now - lastFrameRef.current < 40 || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
        busyRef.current = true
        lastFrameRef.current = now
        try {
          const image = await createImageBitmap(video)
          if (session !== sessionRef.current) { image.close(); return }
          worker.postMessage({ type: 'frame', image, timestamp: now }, [image])
        } catch {
          busyRef.current = false
        }
      }
      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        if (session !== sessionRef.current) return
        const message = event.data
        if (message.type === 'ready') {
          setStatus('searching')
          frameRef.current = requestAnimationFrame(capture)
        } else if (message.type === 'result') {
          busyRef.current = false
          const hand = message.hands?.find((item) => preferredRef.current === 'any' || item.handedness.toLowerCase() === preferredRef.current)
          const intent = hand ? trackerRef.current.update(hand.landmarks, message.timestamp ?? performance.now(), video.videoWidth / Math.max(video.videoHeight, 1), mirrorRef.current) : null
          if (!hand) trackerRef.current.reset()
          intentRef.current = intent
          setUiIntent(intent)
          setStatus(intent ? 'tracking' : 'searching')
        } else {
          stop()
          setStatus('error'); setError('model'); setErrorDetail(message.message ?? null)
        }
      }
      worker.onerror = (event) => { if (session === sessionRef.current) { stop(); setStatus('error'); setError('model'); setErrorDetail(event.message) } }
      worker.postMessage({ type: 'init', assetBase })
    } catch (failure) {
      if (timeoutId) clearTimeout(timeoutId)
      if (session !== sessionRef.current) return
      const kind = failure instanceof DOMException && failure.name === 'TimeoutError' ? 'timeout' : errorKind(failure)
      stop()
      setStatus('error'); setError(kind)
    }
  }, [deviceId, stop])

  useEffect(() => {
    const onHidden = () => { if (document.hidden) stop() }
    document.addEventListener('visibilitychange', onHidden)
    return () => { document.removeEventListener('visibilitychange', onHidden); stop() }
  }, [stop])

  return { videoRef, intentRef, uiIntent, status, error, errorDetail, devices, deviceId, preferredHand, setPreferredHand, mirror, setMirror, sensitivity, setSensitivity, start, stop }
}

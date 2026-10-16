/* Classic worker: MediaPipe 0.10.17 loads its WebAssembly glue with importScripts(). */
self.onmessage = async (event) => {
  const message = event.data
  if (message.type === 'init') {
    try {
      self.exports = {}
      importScripts(`${message.assetBase}/vision_bundle.js`)
      const { FilesetResolver, HandLandmarker } = self.exports
      const files = await FilesetResolver.forVisionTasks(`${message.assetBase}/wasm`)
      self.landmarker = await HandLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: `${message.assetBase}/hand_landmarker.task`, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.55,
        minHandPresenceConfidence: 0.55,
        minTrackingConfidence: 0.5,
      })
      self.postMessage({ type: 'ready' })
    } catch (error) {
      self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
    }
    return
  }

  try {
    if (!self.landmarker) throw new Error('Hand model is not ready')
    const result = self.landmarker.detectForVideo(message.image, message.timestamp)
    self.postMessage({
      type: 'result',
      timestamp: message.timestamp,
      hands: result.landmarks.map((landmarks, index) => ({
        handedness: result.handednesses[index]?.[0]?.categoryName ?? 'Unknown',
        landmarks: landmarks.map(({ x, y, z }) => ({ x, y, z })),
      })),
    })
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  } finally {
    message.image.close()
  }
}

# Local hand tracking assets

- `hand_landmarker.task`: [MediaPipe Hand Landmarker float16 model](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).
- `wasm/*`: copied from `@mediapipe/tasks-vision` version `0.10.17` in `node_modules/@mediapipe/tasks-vision/wasm`.
- `vision_bundle.js`: copied from `@mediapipe/tasks-vision` version `0.10.17` (`vision_bundle.cjs`) for use in a classic Web Worker.

Keep the npm package version and these WebAssembly files in sync. The assets load from this local app; video frames are processed in the browser worker.

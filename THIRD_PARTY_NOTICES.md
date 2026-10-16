# Third-party notices · 第三方声明

Model Atlas itself is released under the [MIT License](LICENSE). This file lists third-party material that is
redistributed with it or bundled into the built app.

本项目自身以 MIT 许可发布。下面列出随项目一起分发、或打包进构建产物的第三方内容及其许可。

## Files shipped in this repository · 仓库中直接分发的文件

### MediaPipe Tasks Vision and Hand Landmarker (Apache License 2.0)

Location: `public/mediapipe/` (see [public/mediapipe/SOURCES.md](public/mediapipe/SOURCES.md) for exact sources)

- `wasm/*` and `vision_bundle.js`: copied from the npm package `@mediapipe/tasks-vision` 0.10.17,
  Copyright Google LLC, licensed under the Apache License, Version 2.0.
- `hand_landmarker.task`: the MediaPipe Hand Landmarker (float16) model published by Google at
  `storage.googleapis.com/mediapipe-models`, licensed under the Apache License, Version 2.0 according to its model card.
- `hand-worker.js` and `SOURCES.md` are part of Model Atlas (MIT).
- `LICENSE` is the Apache License 2.0 text covering the files above.

These files are redistributed unmodified. The full text of the Apache License 2.0 is included as
[public/mediapipe/LICENSE](public/mediapipe/LICENSE) and is copied into the build output next to the files it covers.
It is also available at https://www.apache.org/licenses/LICENSE-2.0.

这些文件未经修改按 Apache-2.0 再分发；许可全文见 [public/mediapipe/LICENSE](public/mediapipe/LICENSE)，构建时会一起复制到产物中。

## Libraries bundled into the built app · 打包进构建产物的依赖

They are installed by `npm install` and are not stored in this repository. Their license texts are in each
package under `node_modules/` and are kept in the build output by Vite where the package provides them.

| Package | License |
| --- | --- |
| react, react-dom | MIT |
| three | MIT |
| @react-three/fiber, @react-three/drei | MIT |
| zustand | MIT |
| monaco-editor, @monaco-editor/react | MIT |
| react-resizable-panels | MIT |
| lucide-react | ISC |
| @mediapipe/tasks-vision | Apache-2.0 |

Run `npm ls --all` to see the full dependency tree, including transitive packages.

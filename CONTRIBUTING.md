# 参与贡献 · Contributing

感谢你愿意改进 Model Atlas！提问、报告问题、补充文档和提交代码都欢迎。参与时请遵守[行为准则](CODE_OF_CONDUCT.md)。

Thanks for helping improve Model Atlas. Questions, bug reports, docs and code are all welcome. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## 准备环境 · Setup

需要 Node.js 22.6 或更新版本。 / Requires Node.js 22.6 or newer.

```bash
npm ci
npm run dev
```

## 提交前请运行 · Before you open a pull request

这些检查和 CI 一致，全部通过后再提交。 / These are the same checks CI runs.

```bash
npm run build          # 类型检查 + 生产构建 / type check and build
npm test               # 解释器、训练流程、学习工具、手势测试 / all test suites
npm run check:graphs   # 内置模型的接口、形状、重叠和穿模检查 / graph integrity
```

## 常见改动 · Common changes

- **让解释器支持新的 PyTorch 写法**：在 `tests/interpreter/cases/` 加一个最小的 `.py` 用例，在装有 PyTorch 的机器上运行 `npm run verify:torch -- --update` 生成期望结果，再修改 `src/core/pytorch/` 直到 `npm run test:interpreter` 通过。
  **Supporting a new PyTorch idiom**: add a minimal case to `tests/interpreter/cases/`, regenerate `tests/interpreter/expected.json` with `npm run verify:torch -- --update` (needs PyTorch), then change `src/core/pytorch/` until `npm run test:interpreter` passes.
- **新增内置模型**：代码放在 `src/data/models/code/`，在 `src/data/models/codeCatalog.ts` 登记，并确认 `npm run check:graphs` 全部为 0。论文或参考实现请写进 [ARCHITECTURE_SOURCES.md](ARCHITECTURE_SOURCES.md)。
  **Adding a built-in model**: put the code in `src/data/models/code/`, register it in `codeCatalog.ts`, keep `npm run check:graphs` clean, and cite sources in `ARCHITECTURE_SOURCES.md`.
- **界面文字**：中英文都要改，文本在 `src/i18n/translations.ts`。 / **UI text**: update both languages in `src/i18n/translations.ts`.

## 提交规范 · Pull requests

- 一个 PR 只做一件事，说明改了什么、为什么改，界面改动请附截图。
- 不要提交 `node_modules/`、`dist/`、日志或个人备份。
- 引入新的第三方文件或依赖时，请确认许可与 MIT 兼容，并更新 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- 提交的代码将以本项目的 [MIT 许可](LICENSE) 发布。

Keep each PR focused, explain what and why, add screenshots for UI changes, and update `THIRD_PARTY_NOTICES.md` when adding third-party files or dependencies. Contributions are released under the project's MIT License.

## 安全问题 · Security

请不要在公开 issue 里报告安全漏洞，做法见 [SECURITY.md](SECURITY.md)。 / Please report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

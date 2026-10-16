# 导入模板 / Import template

Model Atlas 完全在本机运行：3D 化由浏览器里的解释器完成，不联网、不需要 API key、不需要安装 Python，也不会上传或保存你的代码到任何服务器。

## 怎么导入自己的模型

1. **准备文件**：照 `atlas-template/` 的结构写，或者让 AI 帮你改写。
   - 把 [ai-import-prompt.md](ai-import-prompt.md) 整段复制给你常用的 AI 助手（Claude、ChatGPT 等）。在最后粘贴你的代码，AI 会按模板输出 `model.py`、`data.py`、`train.py` 和 `atlas.json`。
   - 应用里“导入模型”对话框的 **复制 AI 提示词** 按钮复制的是同一段内容。
2. **导入**：点“导入模型”，选择这些文件或整个文件夹。
3. **选择**：在列表里勾选要包含的文件，比如不想看训练流程就取消勾选 `train.py`；再选择从哪个文件开始运行。
4. 导入后，代码、3D 结构、参数量都可以继续编辑和查看，改一行就会重新生成。

## 模板里各文件的作用

| 文件 | 必需 | 画出来的内容 |
| --- | --- | --- |
| `model.py` | ✓ | 网络结构。每个 `nn.Module` 类是一个可展开的模块。 |
| `data.py` |  | 一批数据从哪里来。DataLoader 取出的每个张量是图里的一个输入块，形状里带 batch 维。 |
| `train.py` |  | 一个训练步：批数据 → 模型 → 损失 → 反向传播 → 梯度裁剪 → 优化器更新。外层的 epoch / batch 循环只执行一次，“更新参数”模块会写明总共重复多少步。 |
| `atlas.json` |  | 项目名称、说明，以及入口文件 `entry`。 |

只导入 `model.py` 时，看到的是网络本身（推理时的一次前向）。

## 能用和不能用的写法

- **能用**：`torch.nn` 常用层，`torch` / `F` 的常用张量函数，`einops`，`torch.optim` 里的优化器和学习率调度器，`Dataset`、`TensorDataset`、`DataLoader`、`random_split`，`autocast`、`GradScaler`，梯度裁剪。
- **不能用**：第三方模型库（timm、transformers 等）、读取真实文件、依赖真实数值的分支，以及自定义 CUDA 算子。遇到这些情况，会在出错的那一行给出提示。

## 命令行检查（可选）

```bash
node scripts/trace-file.mjs templates/atlas-template/*.py templates/atlas-template/atlas.json
```

这条命令会打印运行顺序、顶层模块、各模块的形状和参数量，适合先确认 AI 改写的文件能被正确追踪。

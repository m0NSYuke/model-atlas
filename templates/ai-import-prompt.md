# 任务：把我的 PyTorch 代码改写成 Model Atlas 导入模板

我要把一个 PyTorch 项目导入 Model Atlas，在本地浏览器里看它的 3D 架构。Model Atlas 自带一个精简的 Python 解释器。它只推算张量形状，不做真实计算，也不读取任何文件。请按下面的模板和规则改写我的代码。

## 输出文件（每个文件放一个独立代码块，代码块上方写出文件名）

| 文件 | 是否必需 | 内容 |
| --- | --- | --- |
| `model.py` | 必需 | 只放网络：所有 `nn.Module` 类和超参数常量。文件末尾在 `if __name__ == "__main__":` 里实例化模型，并用 `torch.randn` / `torch.randint` 造的假输入调用一次。 |
| `data.py` | 可选 | 一个 `Dataset`，它的 `__getitem__` 用假张量返回**一个样本**（形状和真实数据一致），再加一个返回 `DataLoader` 的函数。 |
| `train.py` | 可选 | 损失函数、优化器、学习率调度器和训练循环。用 `from model import ...`、`from data import ...` 引用前两个文件。 |
| `atlas.json` | 可选 | `{"name": "英文名", "zhName": "中文名", "description": "...", "zhDescription": "...", "entry": "train.py"}`。`entry` 是从哪个文件开始运行：想看训练流程就填 `train.py`，只看网络就填 `model.py`。 |

我想包含的部分：【在这里写，例如“模型 + 训练循环”或“只要模型”】

## 必须遵守的规则

1. 只能 import `torch`、`torch.nn`、`torch.nn.functional`、`torch.optim`、`torch.utils.data`、`einops`、`math`、`dataclasses`。`tqdm`、`time`、`random`、`os` 可以保留，但它们在这里不起作用。
2. 不能用 `timm`、`transformers`、`diffusers`、`torchvision` 等第三方库。如果用到了其中的层，请用 `torch.nn` 手写等价实现，并保持参数量和形状一致。
3. 不读写真实文件。删掉 `torch.load`、图片读取、tokenizer 和数据集下载，改成形状相同的假张量（例如图片用 `torch.randn(3, 224, 224)`，类别用 `torch.randint(0, 1000, ())`，token 序列用 `torch.randint(0, vocab, (seq_len,))`）。
4. 所有形状都必须能从常量或配置推算出来。不要写依赖真实数值的控制流，例如 `if loss.item() > 1:`。
5. 训练循环保持原样，包括 epoch 循环、`optimizer.zero_grad()`、`loss.backward()`、梯度裁剪、`optimizer.step()` 和 `scheduler.step()`。Atlas 只画其中一步，并标出总共要重复多少步。
6. 删掉分布式（DDP、`torch.distributed`）、日志上报（wandb、tensorboard）、保存检查点、评估和推理代码。
7. 保持类名、层名和超参数不变，这样 3D 图里的名字能和原代码对上。可以在类的 docstring 里加一行 `zh: 中文说明`，它会显示在模块介绍里。
8. 无法转换的部分，请在回答末尾列出来，并说明原因。

## 我的原始代码

【把你的代码粘贴在这里；有多个文件的话，每个文件前写上文件名】

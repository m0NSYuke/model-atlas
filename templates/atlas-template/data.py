# data.py (optional · 可选): where batches come from.
# Atlas never reads real files. Describe ONE sample's shape with fake tensors (torch.randn / torch.randint);
# the DataLoader then stacks batch_size samples, exactly like real training.
# zh: Atlas 不读取真实文件。用假张量描述“一个样本”的形状即可，DataLoader 会按 batch_size 叠成一批，和真实训练一致。
import torch
from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 64


class FakeImages(Dataset):
    """Stands in for your real dataset: same sample shapes, no real data.
    zh: 代替你的真实数据集：样本形状相同，但没有真实数据。"""

    def __init__(self, size=50_000, image_size=32, num_classes=10):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = torch.randn(3, self.image_size, self.image_size)     # one image · 一张图
        label = torch.randint(0, self.num_classes, ())                # its class id · 类别编号
        return image, label


def make_loader(batch_size=BATCH_SIZE):
    return DataLoader(FakeImages(), batch_size=batch_size, shuffle=True, drop_last=True)

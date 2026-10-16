import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    x = torch.randn(2, 3)
    name = "x"
    print("%s shape: %s" % (name, tuple(x.shape)))
    print("{} {:.2f}".format(name, 1.234), f"{x.shape[0]:>4d}", str(list(x.shape)), repr(name), name.upper(), ",".join(["a", "b"]))
    d = {"a": 1}; d.update(b=2); print(sorted(d.items()), list(reversed(range(0, 10, 3))), sum(i * i for i in range(4)), {i for i in range(3)}, max([1, 5, 2]), divmod(7, 2), 7 // 2, 2 ** 10, abs(-3), round(3.14159, 2), any([0, 1]), all([1, 1]))

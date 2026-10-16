// PyTorch and einops classes that are easiest to describe in Python itself. The interpreter runs this once per
// trace before the user's files; its lines never appear in diagnostics or as source links (see PRELUDE_FILE).

export const PRELUDE_FILE = '<torch>'

export const preludeSource = `
import copy
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


class TransformerEncoder(nn.Module):
    """A stack of num_layers copies of one TransformerEncoderLayer.
    zh: 把同一个 TransformerEncoderLayer 复制 num_layers 份依次堆叠，每层参数各自独立。"""

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal=is_causal)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """A stack of num_layers copies of one TransformerDecoderLayer; every layer also attends to the encoder memory.
    zh: 把同一个 TransformerDecoderLayer 复制 num_layers 份堆叠，每层都会关注编码器输出 memory。"""

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=False):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class Transformer(nn.Module):
    """The encoder-decoder Transformer of "Attention Is All You Need".
    zh: 原始的编码器-解码器 Transformer：编码器读入 src，解码器在生成 tgt 时关注编码结果。"""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation="relu", custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias))
        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, bias)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias))
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, src_is_causal=None, tgt_is_causal=None, memory_is_causal=False):
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

    @staticmethod
    def generate_square_subsequent_mask(sz, device=None, dtype=None):
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


class Rearrange(nn.Module):
    """einops.layers.torch.Rearrange: reorders and regroups axes by a pattern.
    zh: 按 einops 模式重排、合并或拆分维度，不含参数。"""

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_lengths)


class Reduce(nn.Module):
    """einops.layers.torch.Reduce: reduces axes by a pattern (mean, max, sum …).
    zh: 按 einops 模式对某些维度做 mean / max / sum 等归约。"""

    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return reduce(x, self.pattern, self.reduction, **self.axes_lengths)


class ParameterList(nn.Module):
    """A list of learnable tensors registered on the module.
    zh: 一组可学习参数，按下标 0、1、2… 注册在模块上。"""

    def __init__(self, values=None):
        super().__init__()
        self._size = 0
        if values is not None:
            self.extend(values)

    def append(self, value):
        setattr(self, str(self._size), value)
        self._size += 1
        return self

    def extend(self, values):
        for value in values:
            self.append(value)
        return self

    def __getitem__(self, index):
        if index < 0:
            index += self._size
        return getattr(self, str(index))

    def __setitem__(self, index, value):
        setattr(self, str(index), value)

    def __len__(self):
        return self._size

    def __iter__(self):
        return iter([getattr(self, str(i)) for i in range(self._size)])


class ParameterDict(nn.Module):
    """Learnable tensors registered by name.
    zh: 按名字注册的一组可学习参数。"""

    def __init__(self, parameters=None):
        super().__init__()
        self._names = []
        if parameters is not None:
            self.update(parameters)

    def update(self, parameters):
        items = parameters.items() if hasattr(parameters, "items") else parameters
        for key, value in items:
            self[key] = value

    def __setitem__(self, key, value):
        if key not in self._names:
            self._names.append(key)
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return iter(list(self._names))

    def keys(self):
        return list(self._names)

    def values(self):
        return [getattr(self, key) for key in self._names]

    def items(self):
        return [(key, getattr(self, key)) for key in self._names]


def checkpoint(function, *args, use_reentrant=None, context_fn=None, determinism_check="default", debug=False, preserve_rng_state=True, **kwargs):
    return function(*args, **kwargs)


def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    return functions(input)
`

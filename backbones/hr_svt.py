##########################################################
# pytorch v1.6.0
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# April 2021
##########################################################
"""
High Resolution transformer
--using twins-svt
"""
import torch.nn as nn
from .twin_svt import PatchEmbedding, PEG, Transformer, LayerNorm
from .layers import Concatenate, HyperConv2d


activation_dict = {'relu': nn.ReLU(),
                   'relu6': nn.ReLU6(),
                   'prelu': nn.PReLU(),
                   'hardtanh': nn.Hardtanh(),
                   'tanh': nn.Tanh(),
                   'elu': nn.ELU(),
                   'leakyrelu': nn.LeakyReLU(),
                   'selu': nn.SELU(),
                   'gelu': nn.GELU(),
                   'glu': nn.GLU(),
                   # 'swish': nn.SILU(),
                   'sigmoid': nn.Sigmoid(),
                   'hardsigmoid': nn.Hardsigmoid(),
                   'softsign': nn.Softsign(),
                   'softplus': nn.Softplus,
                   'softmin': nn.Softmin(),
                   'softmax': nn.Softmax()}


class Activation(nn.Sequential):
    def __init__(self, activation, inplace=True, modify=True):
        super(Activation, self).__init__()
        act = activation_dict[activation]
        if hasattr(act, 'inplace'):
            act.inplace = inplace
        if hasattr(act, 'min_val'):
            act.min_val = 0 if modify else act.min_val
        self.add_module('activation', act)


class Path4(nn.Sequential):
    def __init__(self, dim, dim_out, patch_size, local_patch_size, depth, peg_kernel_size=3, global_k=7,
                 dropout=0., has_local=True, n_divs=1):
        super().__init__()
        layer = nn.Sequential(
            PatchEmbedding(dim=dim, dim_out=dim_out, patch_size=patch_size, n_divs=n_divs),
            Transformer(dim=dim_out, depth=1, local_patch_size=local_patch_size,
                        global_k=global_k, dropout=dropout, has_local=has_local, n_divs=n_divs),
            PEG(dim=dim_out, kernel_size=peg_kernel_size),  # n_divs=n_divs),
            Transformer(dim=dim_out, depth=depth, local_patch_size=local_patch_size,
                        global_k=global_k, dropout=dropout, has_local=has_local, n_divs=n_divs))

        self.add_module('encoder', layer)
        self.add_module('upsample', nn.Upsample(scale_factor=2))


class Path2(nn.Module):
    def __init__(self, dim, dim_out, dim_side, patch_size, local_patch_size, depth, peg_kernel_size=3, global_k=7,
                 dropout=0., has_local=True, n_divs=1, activation='gelu', norm_layer=LayerNorm):
        super().__init__()
        layer = nn.Sequential(
            PatchEmbedding(dim=dim, dim_out=dim_out, patch_size=patch_size, n_divs=n_divs),
            Transformer(dim=dim_out, depth=1, local_patch_size=local_patch_size,
                        global_k=global_k, dropout=dropout, has_local=has_local, n_divs=n_divs),
            PEG(dim=dim_out, kernel_size=peg_kernel_size),  # n_divs=n_divs),
            Transformer(dim=dim_out, depth=depth, local_patch_size=local_patch_size,
                        global_k=global_k, dropout=dropout, has_local=has_local, n_divs=n_divs))
        self.grp1 = nn.Sequential()
        self.grp1.add_module('encoder', layer)

        self.concat = Concatenate(dim=1, n_divs=n_divs)

        in_channels = dim_out + dim_side
        out_channels = dim_out

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        extra_args = {'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}

        self.grp2 = nn.Sequential()
        self.grp2.add_module('norm', norm_layer(in_channels))
        self.grp2.add_module('act', Activation(activation, inplace=True))
        self.grp2.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False,
                                            **extra_args))
        self.grp2.add_module('upsample', nn.Upsample(scale_factor=2))

    def forward(self, x, x_side):
        x = self.grp1(x)
        x = self.concat([x, x_side])
        x = self.grp2(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class HRBlock(nn.Module):
    def __init__(self, dim, dim_out, local_patch_size, depth, peg_kernel_size=3, global_k=7,
                 dropout=0., has_local=True, n_divs=1, activation='gelu', norm_layer=LayerNorm):
        super().__init__()

        dim_side = dim_out * 2
        self.scale_2 = Path2(dim=dim, dim_out=dim_out, dim_side=dim_side, patch_size=2,
                             local_patch_size=local_patch_size, depth=depth, peg_kernel_size=peg_kernel_size,
                             global_k=global_k, dropout=dropout, has_local=has_local, n_divs=n_divs,
                             activation=activation, norm_layer=LayerNorm)

        self.scale_4 = Path4(dim=dim, dim_out=dim_side, patch_size=4, local_patch_size=local_patch_size,
                             depth=depth, peg_kernel_size=peg_kernel_size, global_k=global_k,
                             dropout=dropout, has_local=has_local, n_divs=n_divs)

        self.concat = Concatenate(dim=1, n_divs=n_divs)

        in_channels = dim + dim_out
        out_channels = dim_out

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        extra_args = {'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}

        self.head = nn.Sequential()
        self.head.add_module('norm', norm_layer(in_channels))
        self.head.add_module('act', Activation(activation, inplace=True))
        self.head.add_module('conv', Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False,
                                            **extra_args))

    def forward(self, x):
        x2 = self.scale_4(x)
        x1 = self.scale_2(x, x2)
        # print(x1.shape, x.shape)
        x_head = self.concat([x1, x])
        x_out = self.head(x_head)
        return x_out


class HRTransformer(nn.Sequential):
    def __init__(self,
                 in_channels=3,
                 img_size=224,
                 local_patch_size=7,
                 depth=1,
                 num_block=2,
                 embed_dim=64,
                 peg_kernel_size=3,
                 global_k=7,
                 dropout=0.,
                 n_divs=1,
                 activation='gelu',
                 norm_layer=LayerNorm):
        super().__init__()
        assert img_size % local_patch_size == 0, f"img_size {img_size} if not divisible by local_patch_size " \
                                                 f"{local_patch_size}"
        dim = in_channels
        dim_out = embed_dim
        for i in range(num_block):
            layer = HRBlock(dim, dim_out, local_patch_size, depth, peg_kernel_size, global_k, dropout,
                            has_local=i < num_block - 1, n_divs=n_divs, activation=activation,
                            norm_layer=norm_layer)
            block = layer if i == 0 else Residual(layer)
            self.add_module(f"block{i + 1}", block)
            dim = embed_dim

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
from timm.models.layers import to_ntuple
from .layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv2d, multiply, dot_product


# #################
# helper methods
# #################
def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# ########
# classes
# #######
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., n_divs=1):
        assert dim % n_divs == 0, f'dim {dim} is not divisible by n_divs {n_divs}'
        super().__init__()
        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        self.net = nn.Sequential(
            Conv2d(dim, dim * mult, 1, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {})),
            nn.GELU(),
            nn.Dropout(dropout),
            Conv2d(dim * mult, dim, 1, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {})),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size, n_divs=1):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size
        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        self.proj = Conv2d(patch_size ** 2 * dim, dim_out, 1, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}))

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        return self.proj(fmap)


class PEG(nn.Module):
    def __init__(self, dim, kernel_size=3):  #, n_divs=1):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
                                       groups=dim, stride=1))
        # # Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        # if n_divs == 1:
        #     self.proj = Residual(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
        #                                    groups=dim, stride=1))
        # else:
        #     self.proj = Residual(HyperConv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2,
        #                                      stride=1, n_divs=n_divs))

    def forward(self, x):
        return self.proj(x)


class LocalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., patch_size=7, n_divs=1):
        super().__init__()
        inner_dim = dim_head * heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.n_divs = n_divs

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        self.to_q = Conv2d(dim, inner_dim, 1, bias=False, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}))
        self.to_kv = Conv2d(dim, inner_dim * 2, 1, bias=False, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}))

        Softmax = nn.Softmax if n_divs == 1 else HyperSoftmax
        self.softmax = nn.Softmax(dim=-1) if n_divs == 1 else HyperSoftmax(dim=0, n_divs=n_divs)
        self.to_out = nn.Sequential(
            Conv2d(inner_dim, dim, 1, **({'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {})),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1=p, p2=p)

        q = self.to_q(fmap)
        k, v = self.to_kv(fmap).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h=h), (q, k, v))

        if self.n_divs == 1:
            dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            # dots = dot_product(q, k, n_divs=self.n_divs, dim=-1)  # now a scalar
            dots = multiply(q, k.transpose(-2, -1), n_divs=self.n_divs, q_dim=-1, v_dim=-2)

        # attn = dots.softmax(dim=-1)
        attn = self.softmax(dots)

        # out = einsum('b i j, b j d -> b i d', attn, v)

        if self.n_divs == 1:
            out = einsum('b i j, b j d -> b i d', attn, v)
        else:
            vs = torch.chunk(v, chunks=self.n_divs, dim=-1)
            out = torch.cat([torch.matmul(attn[i], vs[i]) for i in range(self.n_divs)], dim=-1)

        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h=h, x=x, y=y, p1=p, p2=p)
        return self.to_out(out)


class GlobalAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., k=7, n_divs=1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.n_divs = n_divs

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        extra_args = {'n_divs': n_divs, 'stride': 1} if n_divs > 1 else {}
        self.to_q = Conv2d(dim, inner_dim, 1, bias=False, **extra_args)
        self.to_kv = Conv2d(dim, inner_dim * 2, k, stride=k, bias=False, **({'n_divs': n_divs} if n_divs > 1 else {}))

        self.to_out = nn.Sequential(
            Conv2d(inner_dim, dim, 1, **extra_args),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        q = self.to_q(x)
        k, v = self.to_kv(x).chunk(2, dim=1)

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h=h), (q, k, v))

        if self.n_divs == 1:
            dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            dots = dot_product(q, k, n_divs=self.n_divs, dim=-1)  # now a scalar

        attn = dots.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads=8, dim_head=64, mlp_mult=4, local_patch_size=7, global_k=7, 
                 dropout=0., has_local=True, n_divs=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, LocalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                     patch_size=local_patch_size, n_divs=n_divs)))
                if has_local else nn.Identity(),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout, n_divs=n_divs)))
                if has_local else nn.Identity(),
                Residual(PreNorm(dim, GlobalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                                      k=global_k, n_divs=n_divs))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout, n_divs=n_divs)))
            ]))
            
    def forward(self, x):
        for local_attn, ff1, global_attn, ff2 in self.layers:
            x = local_attn(x)
            x = ff1(x)
            x = global_attn(x)
            x = ff2(x)
        return x


class TwinsSVT_old(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        s2_emb_dim = 128,
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        super().__init__()
        kwargs = dict(locals())

        dim = 3
        layers = []

        for prefix in ('s1', 's2', 's3', 's4'):
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            is_last = prefix == 's4'

            dim_next = config['emb_dim']

            layers.append(nn.Sequential(
                PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
                Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
                PEG(dim = dim_next, kernel_size = peg_kernel_size),
                Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
            ))

            dim = dim_next

        self.layers = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...'),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class TwinsSVT(nn.Module):
    def __init__(
            self,
            img_size=224,
            in_channels=3,

            peg_kernel_size=3,
            dropout=0.,

            patch_size=(4, 2, 2, 2),
            local_patch_size=(7, 7, 7, 7),
            global_k=(7, 7, 7, 7),
            depths=(1, 1, 5, 4),
            embed_dims=(64, 128, 256, 512),
            # mlp_ratio=4,

            # norm_layer=nn.LayerNorm,
            # num_heads=(3, 6, 12, 24),
            # use_checkpoint=False,
            n_divs=1
    ):
        super().__init__()
        self.img_size = img_size
        self.num_layers = len(depths)
        patch_size = to_ntuple(self.num_layers)(patch_size)
        local_patch_size = to_ntuple(self.num_layers)(local_patch_size)
        global_k = to_ntuple(self.num_layers)(global_k)

        assert in_channels % n_divs == 0, f'in_chans [{in_channels}] is not divisible by n_divs [{n_divs}]'
        assert all(t % n_divs == 0 for t in embed_dims), f'embed_dim {embed_dims} is not divisible by n_divs {n_divs}'
        # assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'
        assert all(img_size % np.prod(patch_size) == 0), \
            f'img_size {img_size} is not divisible by patch_size {patch_size} product {np.prod(patch_size)}'

        assert all(img_size % np.prod(local_patch_size[:-1]) == 0), \
            f'img_size {img_size} is not divisible by local_patch_size {local_patch_size[:-1]} product {np.prod(local_patch_size[:-1])}'

        dim = in_channels
        self.layers = nn.ModuleList()

        # for prefix in ('s1', 's2', 's3', 's4'):
        #     config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
        #     is_last = prefix == 's4'
        #     dim_next = config['emb_dim']
        for i_layer in range(self.num_layers):
            dim_next = embed_dims[i_layer]
            is_last = i_layer == self.num_layers - 1
            layer = nn.Sequential(
                PatchEmbedding(dim=dim, dim_out=dim_next, patch_size=patch_size[i_layer], n_divs=n_divs),
                Transformer(dim=dim_next, depth=1, local_patch_size=local_patch_size[i_layer],
                            global_k=global_k[i_layer], dropout=dropout, has_local=not is_last, n_divs=n_divs),
                PEG(dim=dim_next, kernel_size=peg_kernel_size),  # n_divs=n_divs),
                Transformer(dim=dim_next, depth=depths[i_layer], local_patch_size=local_patch_size[i_layer],
                            global_k=global_k[i_layer], dropout=dropout, has_local=not is_last, n_divs=n_divs)
            )

            dim = dim_next
            self.layers.append(layer)
        self.num_features = embed_dims
        # self.layers = nn.Sequential(
        #     *layers,
        #     # nn.AdaptiveAvgPool2d(1),
        #     # Rearrange('... () () -> ...'),
        #     # nn.Linear(dim, num_classes)
        # )

    def forward(self, x):
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        return tuple(outs)
        # return self.layers(x)

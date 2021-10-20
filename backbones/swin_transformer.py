# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# Modified by Alabi Bojesomo, {Copyright (c) Kahlifa University}
# --------------------------------------------------------
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv2d, multiply, dot_product
from functools import partial


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., n_divs=1):
        assert in_features % n_divs == 0, f'in_features [{in_features}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % n_divs == 0, \
            f'hidden_features [{hidden_features}] is not divisible by n_divs [{n_divs}]'
        assert out_features % n_divs == 0, f'out_features [{out_features}] is not divisible by n_divs [{n_divs}]'

        Linear = nn.Linear if n_divs == 1 else HyperLinear

        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = Linear(in_features, hidden_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = Linear(hidden_features, out_features, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.n_divs = n_divs

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.qkv = Linear(dim, dim * 3, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        # Softmax = nn.Softmax if n_divs == 1 else HyperSoftmax
        # # self.softmax = Softmax(dim=-1, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        # self.softmax = nn.Softmax(dim=-1) if n_divs == 1 else HyperSoftmax(dim=0, n_divs=n_divs)
        self.softmax = nn.Softmax(dim=-1)  # the dot product of two hypercomplex returns real, not hypercomplex

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x)
        # print(qkv.shape, (B_, N, 3, self.num_heads, C / self.num_heads))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(qkv.shape)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(q.shape, k.shape, v.shape)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # if self.n_divs == 1:
        #     # attn = (q @ k.transpose(-2, -1))
        #     attn = torch.einsum('bhid,bhjd->bhij', q, k)
        # else:
        #     attn = dot_product(q, k, n_divs=self.n_divs, dim=-1)  # now a scalar
            # attn = multiply(q, k.transpose(-2, -1), n_divs=self.n_divs, q_dim=-1, v_dim=-2)
        # print(attn.shape)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            # if self.n_divs == 1:
            #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            #     attn = attn.view(-1, self.num_heads, N, N)
            # else:
            #     attn = attn.view(self.n_divs, B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            #     attn = attn.view(self.n_divs, -1, self.num_heads, N, N)
            # print(attn.shape)
            # attn = self.softmax(attn)
        # else:
        #     # print(attn.shape, B_, N, C)
        #     attn = self.softmax(attn)
            # print(attn.shape)
        # print(attn.shape)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # print(attn.shape)

        # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # attn is now real (not hypercomplex)
        # print(v.type(), attn.type(), attn.device)
        attn = attn.type_as(v)
        x = torch.einsum('bhij,bhjd->bhid', attn, v)
        x = x.transpose(1, 2).reshape(B_, N, C)
        # if self.n_divs == 1:
        #     # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        #     x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2).reshape(B_, N, C)
        # else:
        #     # x = product(attn, v, n_divs=self.n_divs, dim=-1).transpose(1, 2).reshape(B_, N, C)
        #     vs = torch.chunk(v, chunks=self.n_divs, dim=-1)
        #     x = torch.cat([torch.matmul(attn[i], vs[i]) for i in range(self.n_divs)], dim=-1)\
        #         .transpose(1, 2).reshape(B_, N, C)
        #     # x = multiply(attn, v, n_divs=self.n_divs, q_dim=0, v_dim=-1).transpose(1, 2).reshape(B_, N, C)
        #     # print(x.shape)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, n_divs=n_divs)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):  #  x = checkpoint.checkpoint(blk, x, attn_mask)
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        if self.use_checkpoint:
            attn_windows = checkpoint.checkpoint(self.attn, x_windows, attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.reduction = Linear(4 * dim, 2 * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(4 * dim)
        self.concat = Concatenate(dim=-1, n_divs=n_divs)
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = self.concat([x0, x1, x2, x3])  # B H/2 W/2 4*C

        x = rearrange(x, 'b (h p1) (w p2) c -> b h w (p2 p1 c)', p1=2, p2=2)

        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    """ Patch Expanding Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm, ratio=2, factor=1, n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.dim = dim
        self.ratio = ratio
        self.factor = factor
        Linear = nn.Linear if n_divs == 1 else HyperLinear
        self.expansion = Linear(dim, self.factor * self.ratio * dim, bias=False, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.norm = norm_layer(dim)
        # self.concat = Concatenate(dim=-1, n_divs=n_divs)
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = self.H, self.W

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.norm(x)
        x = self.expansion(x)

        # padding
        # pad_input = (H % self.ratio == 1) or (W % self.ratio == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % self.ratio, 0, H % self.ratio))

        x = x.view(B, H*self.ratio, W*self.ratio, C//self.ratio * self.factor)

        x = rearrange(x, 'b h w (p2 p1 c) -> b (h p1) (w p2) c', p1=self.ratio, p2=self.ratio)

        x = x.view(B, -1, C//self.ratio * self.factor)  # B H*2*W*2 C/2

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.downsample = None
        self.H, self.W, self.Wh, self.Ww = None, None, None, None

    def forward(self, x):  # , H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = self.H, self.W
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
            # if self.use_checkpoint:
            #     x = checkpoint.checkpoint(blk, x, attn_mask)
            # else:
            #     x = blk(x, attn_mask)
        if self.downsample is not None:
            self.downsample.H, self.downsample.W = H, W
            # x_down = self.downsample(x, H, W)
            x_down = self.downsample(x)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            self.Wh, self.Ww = Wh, Ww
            # return x, H, W, x_down, Wh, Ww
            return x, x_down
        else:
            self.Wh, self.Ww = H, W
            # return x, H, W, x, H, W
            return x, x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, n_divs=1):
        super().__init__()
        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        # assert embed_dim % 3 == 0, f'embed_dim [{embed_dim}] is not divisible by 3 (self_attention Q, K, V)'
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        Conv2d = nn.Conv2d if n_divs == 1 else HyperConv2d
        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,
                           **{'n_divs': n_divs for n in [n_divs] if n > 1})
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        """Initialize the weights in backbone.

        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer.H, layer.W = Wh, Ww
            # x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            x_out, x = layer(x)
            H, W, Wh, Ww = layer.H, layer.W, layer.Wh, layer.Ww

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)


class WindowAttentionDecode(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., n_divs=1):
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        super().__init__()
        self.n_divs = n_divs
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        Linear = nn.Linear if n_divs == 1 else HyperLinear
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = Linear(dim, dim * 2, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})  # from skip
        self.q = Linear(dim, dim * 1, bias=qkv_bias, **{'n_divs': n_divs for n in [n_divs] if n > 1})  # from input
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q[0]  # make torchscript happy (cannot use tensor as tuple)

        By, Ny, Cy = y.shape
        # kv = self.kv(x).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(By, Ny, 2, self.num_heads, Cy // self.num_heads).permute(2, 0, 3, 1, 4)  # TODO - coorected
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # if self.n_divs == 1:
        #     # attn = (q @ k.transpose(-2, -1))
        #     attn = torch.einsum('bhid, bhjd-> bhij', q, k)
        # else:
        #     attn = dot_product(q, k, n_divs=self.n_divs, dim=-1)  # now a scalar

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # print(q.shape, k.shape, v.shape, attn.shape, relative_position_bias.unsqueeze(0).shape)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        attn = attn.type_as(v)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # attn is now real (not hypercomplex)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerDecode(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_divs=1, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.norm_mixed = norm_layer(dim)
        self.attn_mixed = WindowAttentionDecode(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, n_divs=n_divs)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       n_divs=n_divs)

        self.H = None
        self.W = None

    def get_window(self, x, B, H, W, C):
        # B, L, C = x.shape
        # H, W = self.H, self.W
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # attn_mask = mask_matrix
        else:
            shifted_x = x
            # attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        return x_windows, (Hp, Wp, pad_r, pad_b)

    def self_attn(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return x

    def mixed_attn(self, x, y, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W

        # B, L, C = x.shape
        # H, W = self.H, self.W
        # x = x.view(B, H, W, C)
        # # y = y.view(B, H, W, C)
        #
        # # pad feature maps to multiples of window size
        # pad_l = pad_t = 0
        # pad_r = (self.window_size - W % self.window_size) % self.window_size
        # pad_b = (self.window_size - H % self.window_size) % self.window_size
        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            # shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            # shifted_x = x
            attn_mask = None

        # # partition windows
        # x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        # x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        x_windows, (Hp, Wp, pad_r, pad_b) = self.get_window(x, B, H, W, C)
        y_windows, _ = self.get_window(y, B, H, W, C)
        # W-MSA/SW-MSA
        attn_windows = self.attn_mixed(x_windows, y_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        return x

    def forward(self, x, y, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: Skip feature, tensor size (B, H1*W1, C/2). H1=2*H
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # self attention
        x = self.self_attn(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        # print(x.shape)

        # attention with skip connection
        shortcut = x
        # x = self.norm_mixed(y)  # TODO - modified incorrectly on 7th July 2021
        x = self.norm_mixed(x)  # TODO - modified correctly on 15th July 2021
        x = self.mixed_attn(x, y, mask_matrix)
        x = shortcut + self.drop_path(x)
        # print(x.shape)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EncodeLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage Encoder.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim//2, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        H, W = self.H, self.W
        if self.downsample is not None:
            self.downsample.H, self.downsample.W = H, W
            # x = self.downsample(x, H, W)
            x = self.downsample(x)
            H, W = (H + 1) // 2, (W + 1) // 2

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        # if self.downsample is not None:
        #     x = self.downsample(x, H, W)
        #     H, W = (H + 1) // 2, (W + 1) // 2
        self.H, self.W = H, W
        # return x, H, W
        return x


class DecodeLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage Encoder.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the beginning of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()
        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch expanding layer
        if upsample is not None:
            self.upsample = upsample(dim=dim*2, norm_layer=norm_layer, n_divs=n_divs)
        else:
            self.upsample = None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            layer = SwinTransformerDecode(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            self.blocks.append(layer)
            self.H, self.W, self.H1, self.W1 = None, None, None, None

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: skip feature, tensor size (B, H1*W1, C).
            H, W: Spatial resolution of the input feature.
            H1, W1: Spatial resolution of the skip feature.
        """
        H, W, H1, W1 = self.H, self.W, self.H1, self.W1
        B, _, _ = x.shape
        # print(H, W, H1, W1)
        if self.upsample is not None:
            self.upsample.H, self.upsample.W = H, W
            # x = self.upsample(x, H, W)
            x = self.upsample(x)
            H, W = H * 2, W * 2

        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
            H, W = H1, W1

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y, attn_mask)
            else:
                x = blk(x, y, attn_mask)
                # if i == 0:
                #     x = blk(x, y, attn_mask)
                # else:
                #     x = blk(x, attn_mask)
        self.H, self.W = H, W
        # return x, H, W
        return x


class DecodeLayerUnet(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 upsample=None,
                 use_checkpoint=False,
                 n_divs=1,
                 merge_type='concat'):
        super().__init__()
        assert merge_type in ['concat', 'add']

        assert dim % n_divs == 0, f'dim [{dim}] is not divisible by n_divs [{n_divs}]'
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.upsample = upsample
        if self.upsample is not None:
            self.upsample = upsample(dim=dim*2, norm_layer=norm_layer, n_divs=n_divs)

        self.merge_type = merge_type
        if self.merge_type == 'concat':
            self.concat = Concatenate(dim=-1, n_divs=n_divs)
            Linear = nn.Linear if n_divs == 1 else HyperLinear
            self.fc = Linear(2*dim, dim, **{'n_divs': n_divs for n in [n_divs] if n > 1})
            # self.merge = nn.Sequential(concat, fc)

        # print(depth)
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                n_divs=n_divs,
                use_checkpoint=use_checkpoint
            )
            for i in range(depth)])
        self.H, self.W, self.H1, self.W1 = None, None, None, None

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            y: Input feature, tensor size (B, H1*W1, C).
        """
        H, W, H1, W1 = self.H, self.W, self.H1, self.W1
        B, _, _ = x.shape
        # print(H, W, H1, W1)
        if self.upsample is not None:
            self.upsample.H, self.upsample.W = H, W
            # x = self.upsample(x, H, W)
            x = self.upsample(x)
            H, W = H * 2, W * 2

        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
            H, W = H1, W1

        if self.merge_type == 'concat':
            x = self.concat([x, y])
            x = self.fc(x)
        else:
            x = x + y

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, y, attn_mask)
            else:
                x = blk(x, y, attn_mask)
        self.H, self.W = H, W
        return x


class Head(nn.Module):
    def __init__(self, decode_dim=96, patch_size=4, out_chans=None, img_size=(224,), n_divs=1):
        super().__init__()
        assert decode_dim % n_divs == 0, f'dim [{decode_dim}] is not divisible by n_divs [{n_divs}]'
        self.expand = PatchExpanding(decode_dim, ratio=patch_size, factor=patch_size, n_divs=n_divs)
        self.final = nn.Sequential(
            nn.Linear(in_features=decode_dim, out_features=out_chans if out_chans else decode_dim),
            nn.Sigmoid()
        )
        self.out_chans = out_chans if out_chans else decode_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        H, W = self.H, self.W
        B, _, _ = x.shape
        H1, W1 = self.img_size
        self.expand.H, self.expand.W = H, W
        # x = self.expand(x, H, W)
        x = self.expand(x)
        H, W = H * self.patch_size, W * self.patch_size
        if H1 != H or W1 != W:
            x = x.view(B, H, W, -1)
            x = x[:, :H1, :W1, :].contiguous()
            x = x.view(B, H1 * W1, -1)
        # print(x.shape)
        x = self.final(x)
        return x


class LearnVectorBlock(nn.Module):
    def __init__(self, in_channels, featmaps, filter_size, act_layer=nn.GELU):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = featmaps

        padding = (filter_size[0] // 2, filter_size[0] // 2)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=featmaps, kernel_size=filter_size, padding=padding),
            act_layer()
        )

    def forward(self, x):
        x = self.layer(x)
        return x

    def extra_repr(self) -> str:
        extra_str = f"in_channels={self.in_channels}, out_channels={self.out_channels}, activation={self.activation}"
        return extra_str


class SwinEncoderDecoderTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 img_size=(224,),
                 patch_size=4,
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 n_divs=1):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=n_chans, kernel_size=3, padding=1),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = EncodeLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # add a norm layer for each output
        # for i_layer in out_indices:
        #     layer = norm_layer(num_features[i_layer])
        #     layer_name = f'norm{i_layer}'
        #     self.add_module(layer_name, layer)

        # build up layers
        self.uplayers = nn.ModuleList()
        self.decode_dim = num_features[::-1]
        for i_layer in range(self.num_layers):
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayer(
                dim=decode_dim,
                depth=depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer != 0 else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, patch_size=self.patch_size, out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        H, W = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(H, W), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        features = []
        hw = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            layer.H, layer.W = H, W
            # x, H, W = layer(x, H, W)
            x = layer(x)
            H, W = layer.H, layer.W
            # print(i, H, W, x.shape)
            features.insert(0, x)
            hw.insert(0, (H, W))

        for i in range(self.num_layers):
            layer = self.uplayers[i]
            y = features[i]
            H1, W1 = hw[i]
            # print(i, H, W, H1, W1)
            layer.H, layer.W, layer.H1, layer.W1 = H, W, H1, W1
            # x, H, W = layer(x, y, H, W, H1, W1)
            x = layer(x, y)
            H, W = layer.H, layer.W
            # if i == 0:
            #     y = features[i]
            #     x, H, W = layer(x, y, H, W)
            # else:
            #     x, H, W = layer(x, H, W)
        self.head.H, self.head.W = H, W
        # x = self.head(x, H, W)
        x = self.head(x)
        H1, W1 = to_2tuple(self.img_size)
        # x = x.view(-1, H*self.patch_size, W*self.patch_size, self.head.out_chans).permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, H1, W1, self.head.out_chans).permute(0, 3, 1, 2).contiguous()

        return x


class SwinUNet(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int/tuple): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(self,
                 img_size=(256,),
                 patch_size=4,
                 in_chans=3,
                 out_chans=None,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 decode_depth=None,
                 num_heads=(3, 6, 12, 24),
                 window_size=(1, 7, 7),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 n_divs=1,
                 merge_type='concat'):
        super().__init__()

        # assert in_chans % n_divs == 0, f'in_chans [{in_chans}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % n_divs == 0, f'embed_dim [{embed_dim}] is not divisible by n_divs [{n_divs}]'
        assert embed_dim % num_heads[0] == 0, f'embed_dim [{embed_dim}] is not divisible by num_head[0]=={num_heads[0]}'

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        n_chans = int(np.ceil(in_chans / n_divs) * n_divs)
        self.learn_vector = nn.Sequential(
            nn.Conv3d(in_channels=in_chans, out_channels=n_chans, kernel_size=(1, 3, 3),
                      padding=(0, 1, 1)),
            nn.GELU()
        ) if n_chans != in_chans else nn.Identity()

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=n_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            n_divs=n_divs
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        assert np.all(k % v for k, v in zip(num_features, num_heads)), 'embed_dim/ num_features not a multiple of heads'
        self.num_features = num_features

        # build up layers
        self.uplayers = nn.ModuleList()
        self.decode_dim = num_features[::-1]

        if decode_depth is None:
            decode_depths = depths
        else:
            decode_depths = [decode_depth] * self.num_layers

        for i_layer in range(self.num_layers):
            # print(i_layer)
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayerUnet(
                dim=decode_dim,
                depth=decode_depths[i_layer],  # 1,  #depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer != 0 else None,
                use_checkpoint=use_checkpoint,
                n_divs=n_divs,
                merge_type=merge_type
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, ratio=self.patch_size[-1], out_chans=out_chans, n_divs=n_divs,
                         img_size=self.img_size, in_depth=in_depth, out_depth=out_depth)

    def init_weights(self):
        """Initialize the weights in backbone.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

    def forward(self, x):
        """Forward function."""
        x = self.learn_vector(x)
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            x = layer(x)
            features.insert(0, x)

        for layer, y in zip(self.uplayers, features):
            # print(x.shape, y.shape)
            x = layer(x, y)

        x = self.head(x)

        return x


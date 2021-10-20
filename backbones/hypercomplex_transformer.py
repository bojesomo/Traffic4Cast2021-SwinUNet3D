# --------------------------------------------------------
# Rethinking Transformer Attention with Hypercomplex Linear
# Copyright (c) 2021 Khalifa University
# Written by Alabi Bojesomo
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, to_ntuple, trunc_normal_
# from .layers import HyperLinear, HyperSoftmax, Concatenate, HyperConv2d, multiply, dot_product
from fast_hypercomplex import QuaternionLinear, SedenionLinear
from einops.layers.torch import Reduce, Rearrange
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange, reduce
from functools import partial


def get_window(x, window_size):
    """
    window_size = int
    """
    if x.ndim == 4:  # b c h w
        x = rearrange(x, 'b c (h s1) (w s2) -> b h w (s1 s2 c)', s1=window_size, s2=window_size)
    else:  # b c d h w
        # Todo - to deal with 3D later
        x = rearrange(x, 'b c d (h s1) (w s2) -> b d h w (s1 s2 c)', s1=window_size, s2=window_size)
    return x


def reverse_window(x, window_size):
    """
    window_size = int
    """
    if x.ndim == 4:  # b c h w
        x = rearrange(x, 'b h w (s1 s2 c)  -> b c (h s1) (w s2)', s1=window_size, s2=window_size)
    else:  # b c d h w
        # Todo - to deal with 3D later
        x = rearrange(x, 'b d h w (s1 s2 c) -> b c d (h s1) (w s2)', s1=window_size, s2=window_size)
    return x


def img2patch(img, p1, p2=None):
    p2 = p2 or p1
    x = rearrange(img, 'b c (p1 h) (p2 w) -> b h w (p1 p2 c)', p1=p1, p2=p2)
    return x


def patch2img(img, p1, p2=None):
    p2 = p2 or p1
    x = rearrange(img, 'b h w  (p1 p2 c) -> b c (p1 h) (p2 w)', p1=p1, p2=p2)
    return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, out_size=None, norm_layer=nn.LayerNorm):
        super().__init__()
        out_size = out_size or (2 * dim)
        # assert (out_size % 4 == 0), f"number of output features {out_size} must be divisible by 4"
        self.out_size = out_size
        # self.layer = QuaternionLinear(in_features=4*dim, out_features=self.out_size)
        # self.linear = nn.Linear(in_features=self.out_size, out_features=self.out_size)
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4*dim, out_features=self.out_size, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2))

        x = get_window(x, window_size=2)
        # x = self.layer(x)
        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, ratio=2, factor=1):
        super().__init__()

        self.dim = dim
        self.ratio = ratio
        self.factor = factor
        out_size = self.factor * self.ratio * dim
        self.norm = norm_layer(dim)
        self.expand = nn.Linear(in_features=dim, out_features=out_size, bias=False)

    def forward(self, x):
        # x = get_window(x, window_size=2)
        # x = self.layer(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = self.expand(x)
        x = reverse_window(x, window_size=self.ratio)
        # if x.ndim == 4:
        #     x = rearrange(x, 'b h w c -> b c h w')
        # else:
        #     x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class HyperAttention(nn.Module):
    def __init__(self, dim, attn_bias=False, attn_drop=0., proj_drop=0., act_layer=nn.GELU):
        super().__init__()
        hidden_dim = 16 * dim
        self.short_range = SedenionLinear(in_features=hidden_dim, out_features=hidden_dim, bias=attn_bias)
        # self.long_range = SedenionLinear(in_features=hidden_dim, out_features=hidden_dim, bias=attn_bias)
        self.act = act_layer()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        x1 = get_window(x, window_size=4)
        # x2 = img2patch(x, p1=4, p2=4)

        x1 = self.short_range(x1)
        # x2 = self.long_range(x2)

        x1 = reverse_window(x1, window_size=4)
        # x2 = patch2img(x2, p1=4, p2=4)

        y = x1  # + x2
        y = rearrange(y, 'b c h w -> b h w c')
        y = self.act(y)

        y = self.attn_drop(y)
        y = self.proj(y)
        y = self.proj_drop(y)
        y = rearrange(y, 'b h w c -> b c h w')
        return y


class HyperTransformerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, attn_bias=False,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.attn_bias = attn_bias

        self.norm1 = norm_layer(dim)
        self.attn = HyperAttention(dim, attn_bias=attn_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.use_checkpoint = use_checkpoint

    def attn_forward(self, x):
        # shortcut = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm1(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.attn(x)
        # x = shortcut + x
        return x

    def mlp_forward(self, x):
        # shortcut = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm2(x)
        x = self.mlp(x)
        # x = self.drop_path(x)
        x = rearrange(x, 'b h w c -> b c h w')
        # x = shortcut + x

        return x

    def forward(self, x):
        B, C, H, W = x.shape
        # padding
        pad_input = (H % 4 != 0) or (W % 4 != 0)
        pad_h = 0 if H % 4 == 0 else 4 - H % 4
        pad_w = 0 if W % 4 == 0 else 4 - W % 4
        # print(f"x before: {x.shape}, pad: {pad_input}")
        if pad_input:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.attn_forward, x)
        else:
            x = self.attn_forward(x)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.mlp_forward, x)
        else:
            x = x + self.mlp_forward(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4, attn_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, constant_dim=False,
                 downsample=False, use_checkpoint=False):
        super().__init__()

        if isinstance(drop_path, int):
            drop_path = to_ntuple(depth)(drop_path)

        # patch merging layer
        self.constant_dim = constant_dim
        in_dim = dim if self.constant_dim else dim // 2
        self.downsample = PatchMerging(dim=in_dim, out_size=dim) if downsample else nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            HyperTransformerBlock(
                dim=dim,  # hidden_dim,
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        # # patch merging layer
        # self.downsample = PatchMerging(dim=dim, out_size=2 * dim) if downsample else nn.Identity()

    def forward(self, x):
        """

        :param x: Input feature, tensor size (B, C, H, W)
        :return:
        """
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        # x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
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


class HyperTransformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96,  patch_size=4, patch_norm=True,
                 depths=(2, 2, 6, 2), mlp_ratio=4, attn_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_checkpoint=False,
                 num_classes=10, extract_features=False):
        super().__init__()
        self.num_layers = len(depths)
        # embed_dim = in_channels
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.extract_features = extract_features

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.num_features = []
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            present_dim = int(embed_dim * 2 ** i_layer)
            layer = BasicLayer(
                dim=present_dim,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=(i_layer != 0),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            self.num_features.append(present_dim)
            norm = norm_layer(present_dim)
            self.norms.append(norm)

        if not self.extract_features:
            self.classifier = nn.Linear(present_dim, num_classes)

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
        x = self.patch_embed(x)

        outs = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # print(i, norm, x.shape)
            x = layer(x.contiguous())
            x = rearrange(x, 'b c h w -> b h w c')
            x = norm(x)
            x = rearrange(x, 'b h w c -> b c h w')
            outs.append(x)
        if self.extract_features:
            outs = tuple(outs)
        else:
            # x = nn.AdaptiveAvgPool2d(1)(x)
            x = reduce(x, 'b c h w -> b c', 'mean')
            outs = self.classifier(x)
        return outs


class DecodeLayer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4, attn_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, constant_dim=False,
                 upsample=False, use_checkpoint=False, merge_type='concat'):
        super().__init__()
        assert merge_type in ['concat', 'add', 'both']
        if isinstance(drop_path, int):
            drop_path = to_ntuple(depth)(drop_path)

        # patch merging layer
        self.constant_dim = constant_dim
        in_dim = dim if self.constant_dim else dim * 2
        factor = 2 if self.constant_dim else 1
        self.upsample = PatchExpanding(dim=in_dim, factor=factor, norm_layer=norm_layer) if upsample else nn.Identity()

        self.merge_type = merge_type
        if self.merge_type in ['concat', 'both']:
            self.concat = partial(torch.cat, dim=1)
            self.fc = nn.Linear(2*dim, dim, bias=False)

        # build blocks
        self.blocks = nn.ModuleList([
            HyperTransformerBlock(
                dim=dim,  # hidden_dim,
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        # # patch merging layer
        # self.downsample = PatchMerging(dim=dim, out_size=2 * dim) if downsample else nn.Identity()

    def forward(self, x, y):
        """

        :param x: Input feature, tensor size (B, C, H, W)
        :return:
        """
        _, _, Hx, Wx = x.shape
        _, _, Hy, Wy = y.shape

        x = self.upsample(x)
        if Hx != Hy or Wx != Wy:
            x = x[:, :, :Hy, :Wy]

        if self.merge_type in ['concat', 'both']:
            x = self.concat([x, y])
            x = rearrange(x, 'b c h w -> b h w c')
            x = self.fc(x)
            x = rearrange(x, 'b h w c -> b c h w')
        if self.merge_type in ['add', 'both']:
            x = x + y

        for blk in self.blocks:
            x = blk(x)

        return x


class Head(nn.Module):
    def __init__(self, decode_dim=96, patch_size=4, out_chans=None, img_size=(224,),
                 with_sigmoid=True):
        super().__init__()
        self.expand = PatchExpanding(decode_dim, ratio=patch_size, factor=patch_size)
        self.final = nn.Sequential(
            nn.Linear(in_features=decode_dim, out_features=out_chans if out_chans else decode_dim),
            nn.Sigmoid() if with_sigmoid else nn.Identity()
        )
        self.out_chans = out_chans if out_chans else decode_dim
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.H, self.W = None, None

    def forward(self, x):  # , H, W):
        H1, W1 = self.img_size
        x = self.expand(x)
        _, _, H, W = x.shape
        if H1 != H or W1 != W:
            x = x[:, :, :H1, :W1].contiguous()
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.final(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x


class HyperTransformerUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, embed_dim=96, img_size=(256, 256),
                 patch_size=4, patch_norm=True, constant_dim=False,
                 depths=(2, 2, 6, 2), decode_depth=None, mlp_ratio=4, attn_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_checkpoint=False, with_sigmoid=True,
                 merge_type='concat', use_neck=False,):
        super().__init__()
        self.num_layers = len(depths)
        # embed_dim = in_channels
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.img_size = to_2tuple(img_size)
        self.merge_type = merge_type
        self.use_neck = use_neck
        self.constant_dim = constant_dim

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.num_features = []
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            present_dim = embed_dim if self.constant_dim else int(embed_dim * 2 ** i_layer)
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            layer = BasicLayer(
                dim=present_dim,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=(i_layer != 0),
                use_checkpoint=use_checkpoint,
                constant_dim=self.constant_dim,
            )
            self.layers.append(layer)
            self.num_features.append(present_dim)
            # norm = norm_layer(present_dim)
            # self.norms.append(norm)

        # build neck layer
        if self.use_neck:
            self.neck = BasicLayer(
                dim=present_dim,
                depth=2,
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=False,
                use_checkpoint=use_checkpoint,
            )

        # build up layers
        self.decode_dim = self.num_features[::-1]
        if decode_depth is None:
            decode_depths = depths
        else:
            decode_depths = [decode_depth] * self.num_layers
        self.uplayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])]
            decode_dim = self.decode_dim[i_layer]
            layer = DecodeLayer(
                dim=decode_dim,
                depth=decode_depths[i_layer],
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                act_layer=act_layer,
                upsample=(i_layer != 0),
                use_checkpoint=use_checkpoint,
                merge_type=merge_type,
                constant_dim=constant_dim,
            )
            self.uplayers.append(layer)

        self.head = Head(decode_dim=decode_dim, patch_size=self.patch_size, out_chans=out_channels,
                         img_size=self.img_size, with_sigmoid=with_sigmoid)

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
        x = self.patch_embed(x)

        features = []
        for i, layer in enumerate(self.layers):
            # print(i, norm, x.shape)
            x = layer(x)  # .contiguous())
            features.insert(0, x)
            # print(f"left: {i} == {x.shape}")

        if self.use_neck:
            x = self.neck(x)
            # print(f"neck: == {x.shape}")

        for layer, y in zip(self.uplayers, features):
            # print(x.shape, y.shape)
            x = layer(x, y)

        x = self.head(x)
        return x


class EncodeLayer(nn.Module):
    def __init__(self, dim, depth, dim_out=None, mlp_ratio=4, attn_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 downsample=False, use_checkpoint=False):
        super().__init__()

        if isinstance(drop_path, int):
            drop_path = to_ntuple(depth)(drop_path)

        # patch merging layer
        # hidden_dim = 2 * dim
        if dim_out is None:
            dim_out = dim
        self.downsample = PatchMerging(dim=dim, out_size=dim_out) if downsample else nn.Identity()

        # build blocks
        self.blocks = nn.ModuleList([
            HyperTransformerBlock(
                dim=dim,  # hidden_dim,
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)
        ])

        # # patch merging layer
        # self.downsample = PatchMerging(dim=dim, out_size=2 * dim) if downsample else nn.Identity()

    def forward(self, x):
        """

        :param x: Input feature, tensor size (B, C, H, W)
        :return:
        """
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        # x = self.downsample(x)
        return x


class Transformer(nn.Module):
    def __init__(self, in_channels=3, embed_dim=96,  patch_size=4, patch_norm=True,
                 num_classes=10,
                 depths=(2, 2, 6, 2), mlp_ratio=4, attn_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_checkpoint=False, extract_features=False):
        super().__init__()
        self.num_layers = len(depths)
        # embed_dim = in_channels
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.extract_features = extract_features

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        self.num_features = []
        self.norms = nn.ModuleList()
        for i_layer in range(self.num_layers):
            present_dim = embed_dim  #int(embed_dim * 2 ** i_layer)
            layer = EncodeLayer(
                dim=present_dim,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                attn_bias=attn_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                act_layer=act_layer,
                downsample=(i_layer != 0),
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            self.num_features.append(present_dim)
            norm = norm_layer(present_dim)
            self.norms.append(norm)

            if not self.extract_features:
                self.classifier = nn.Linear(embed_dim, num_classes)

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
        x = self.patch_embed(x)

        outs = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # print(i, norm, x.shape)
            x = layer(x.contiguous())
            x = rearrange(x, 'b c h w -> b h w c')
            x = norm(x)
            x = rearrange(x, 'b h w c -> b c h w')
            outs.append(x)

        if self.extract_features:
            outs = tuple(outs)
        else:
            # x = nn.AdaptiveAvgPool2d(1)(x)
            x = reduce(x, 'b c h w -> b c', 'mean')
            outs = self.classifier(x)

        return outs

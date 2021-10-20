##########################################################
# Alabi Bojesomo
# Khalifa University
# Abu Dhabi, UAE
# June 2021
##########################################################

import os
import sys
sys.path.extend(os.getcwd())
from torch.nn import functional as F
from torch import nn, optim
from torch.nn import Conv2d as RealConv2d

from fast_hypercomplex import (ComplexConv2d, QuaternionConv2d, OctonionConv2d, SedenionConv2d, Concatenate,
                               ComplexDropout2d, QuaternionDropout2d, OctonionDropout2d, SedenionDropout2d)
from fast_hypercomplex.ops import hypercomplex_sigmoid, hypercomplex_silu, hypercomplex_tanh

import numpy as np
# from torch import nn, optim
import warnings
#from Traffic4Cast2021.backbone.swin_transformer import SwinTransformer, SwinEncoderDecoderTransformer
#from Traffic4Cast2021.backbone.swin_transformer3d import (SwinTransformer3D, SwinEncoderDecoderTransformer3D,
#                                                          SwinUNet3D, SwinUPerNet3D)
from Traffic4Cast2021.backbone.resnet import HyperUnet as ResNetUnet, Encoder, DecodeBlock
from backbones.swin_transformer import SwinTransformer, SwinEncoderDecoderTransformer
from backbones.swin_transformer3d import (SwinTransformer3D, SwinEncoderDecoderTransformer3D,
                                                          SwinUNet3D, SwinUPerNet3D)
from backbones.hypercomplex_transformer import HyperTransformerUNet
from Traffic4Cast2021.others.sungbinchoi.models import Net as DenseUnet
# from deepspeed.ops.adam.cpu_adam import DeepSpeedCPUAdam
from functools import partial


ROW_AXIS = 2
COL_AXIS = 3
CHANNEL_AXIS = 1
BATCH_AXIS = 0
criterion_dict = {'mae': nn.L1Loss(),
                  'mse': nn.MSELoss(),
                  'bce': nn.BCELoss(),
                  'binary_crossentropy': nn.BCELoss(),
                  'categorical_crossentropy': nn.CrossEntropyLoss(),
                  }
conv_dict = {'sedenion': SedenionConv2d,
             'octonion': OctonionConv2d,
             'quaternion': QuaternionConv2d,
             'complex': ComplexConv2d,
             'real': RealConv2d}
n_div_dict = {'sedenion': 16,
              'octonion': 8,
              'quaternion': 4,
              'complex': 2,
              'real': 1}


def get_extras(depth, stages, activation):  # , n_divs=1):
    num_blocks = [depth] * stages
    act_layer = {'relu': F.relu,
                 'prelu': F.prelu,
                 'hardtanh': F.hardtanh,
                 'tanh': F.tanh,
                 'elu': F.elu,
                 'leakyrelu': F.leaky_relu,
                 'selu': F.selu,
                 'gelu': F.gelu,
                 'silu': F.silu,
                 'sigmoid': F.sigmoid,
                 }[activation]
    return num_blocks, act_layer


class HyperResNetUnet(ResNetUnet):
    def __init__(self, in_channels, n_classes, start_filters, stages=3, depth=3,
                 # use_group_norm=False, blk_type='resnet', memory_efficient=True, classifier='sigmoid',
                 net_type='sedenion', n_divs=None, activation='relu', inplace_activation=False, constant_dim=False,
                 use_se=False,
                 **kwargs):
        self.kwargs = kwargs
        # Details here
        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        self.n_divs = n_divs

        in_planes = int(n_divs * np.ceil(start_filters / n_divs))
        if start_filters < in_planes:
            warnings.warn(f"start_filters = {start_filters} < in_planes used [{in_planes}]")

        num_blocks, act_layer = get_extras(depth, stages, activation)  # , n_divs=n_divs)
        super().__init__(num_blocks=num_blocks,
                         in_channels=in_channels,
                         num_classes=n_classes,
                         in_planes=in_planes,
                         n_divs=n_divs,
                         act_layer=act_layer,
                         inplace_activation=inplace_activation,
                         constant_dim=constant_dim,
                         use_se=use_se,
                         )


class HyperDenseNetUnet(DenseUnet):
    def __init__(self, in_channels, n_classes, start_filters, stages=3, depth=3,
                 net_type='sedenion', n_divs=None, dense_type='A',
                 use_se=False,
                 **kwargs):
        self.kwargs = kwargs
        # Details here
        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        self.n_divs = n_divs

        in_planes = int(n_divs * np.ceil(start_filters / n_divs))
        if start_filters < in_planes:
            warnings.warn(f"start_filters = {start_filters} < in_planes used [{in_planes}]")

        super().__init__(nb_layers=tuple([depth] * stages),
                         input_channels=in_channels,
                         out_size=n_classes,
                         encode_dims=tuple([min(in_planes + 32*i, 128) for i in range(stages)]), #(64, 96, 128, 128, 128, 128, 128, 128),
                         n_divs=n_divs,
                         dense_type=dense_type,
                         use_se=use_se,
                         )


class HyperSwinTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Details here
        args.convArgs = {"padding": "same", "bias": False, "weight_init": 'hypercomplex'}
        if args.net_type.lower() == 'real':
            args.convArgs.pop('weight_init')
        args.bnArgs = {"momentum": 0.9, "eps": 1e-04}
        args.actArgs = {"activation": args.hidden_activation}


        args.n_divs = n_div_dict[args.net_type.lower()]
        self.n_divs = args.n_divs
        # print(args)
        h, w = args.height, args.width

        frame_shape = (h, w)

        ##########################
        self.net_type = args.net_type
        self.input_shape = (None, args.in_channels, *frame_shape)

        if hasattr(args, 'n_divs'):
            n_divs = args.n_divs
        else:
            n_divs = n_div_dict[args.net_type.lower()]

        n_multiples_in = int(n_divs * np.ceil(args.in_channels / n_divs))
        n_multiples_out = args.n_classes  # args.n_frame_out * args.n_channels_out

        self.n_stages = args.stages

        # Stage 1 - Vector learning and preparation
        if n_divs > 1:
            # print(n_divs)
            self.z0_shape = (None, n_multiples_in, *frame_shape)
            # TODO - change kernel size to 3 (May 11, 2021)
            encode_vector = LearnVectorBlock(args, self.input_shape, n_multiples_in, (3, 3), block_i=1)
        else:
            self.z0_shape = self.input_shape
            encode_vector = nn.Identity()

        patch_size = args.patch_size if hasattr(args, 'patch_size') else 4
        # num_heads = tuple([3 * (k + 1) for k in range(self.n_stages)])  # TODO - modified on May19,2021
        num_heads = tuple([8 for _ in range(self.n_stages)])  # Revisiting Vision Transformer
        heads_ = np.lcm.reduce(num_heads)
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if args.sf > embed_dim:
            embed_dim = int(embed_dim * np.ceil(args.sf / embed_dim))
        net = SwinTransformer(img_size=frame_shape,
                              use_checkpoint=args.memory_efficient if hasattr(args, 'memory_efficient')
                              else False,
                              patch_size=patch_size,
                              depths=tuple([args.nb_layers] * self.n_stages),
                              num_heads=num_heads,
                              out_indices=tuple([k for k in range(self.n_stages)]),
                              # embed_dim=int(n_divs * heads_ * np.ceil(n_multiples_out / (n_divs * heads_))),
                              embed_dim=embed_dim,
                              in_chans=n_multiples_in,
                              n_divs=n_divs,
                              mlp_ratio=4,
                              ape=True)

        self.encode_vector = encode_vector
        self.features = net

        num_features = net.num_features

        # code
        self.z_code = CreateConvBnLayer(args, [None, num_features[-1], None, None], num_features[-1] * 2, layer_i=100)

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            sf_i = num_features[i-1]
            x_shape = self.z_code.output_shape if i == self.n_stages else eval(f'self.z_dec{i + 1}.output_shape')
            y_shape = [None, sf_i, None, None]  # eval(f'self.z_enc{i}.output_shape')
            dec = DecodeBlock(args, x_shape, y_shape, sf_i, layer_i=100 + i,
                              scale_factor=1 if i == self.n_stages else 2)
            setattr(self, f'z_dec{i}', dec)
        self.z_dec0 = DecodeBlock(args, self.z_dec1.output_shape, self.z0_shape, n_multiples_out, layer_i=200,
                                  scale_factor=patch_size)

        # TODO - change kernel size to 3 (May 11, 2021)
        self.decode_vector = LearnVectorBlock(args, self.z_dec0.output_shape, n_multiples_out, (3, 3))
        self.classifier = Activation(args, args.classifier_activation)

    def forward(self, x):  # x is [dynamic, static]

        z0 = self.encode_vector(x)

        features = self.features(z0)

        # code
        z100 = self.z_code(features[-1])

        # Stage 3 - Decoder
        for i in range(self.n_stages, 0, -1):
            z_ = eval(f'z100 if i == self.n_stages else z10{i + 1}')
            exec(f'z10{i} = self.z_dec{i}([z_, features[{i-1}]])')

        z200 = eval(f'self.z_dec0([z101, z0])')
        # print(z200.shape)
        model_output = self.classifier(self.decode_vector(z200))
        return model_output


class HyperSwinEncoderDecoder(SwinEncoderDecoderTransformer):
    def __init__(self, in_channels, n_classes, start_filters, stages=3, depth=3,
                 # use_group_norm=False, blk_type='resnet', memory_efficient=True, classifier='sigmoid',
                 net_type='sedenion', n_divs=None, activation='relu', height=495, width=436,
                 patch_size=4, head=8,
                 **kwargs):

        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        heads_ = head
        n_multiples_in = int(n_divs * np.ceil(in_channels / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if start_filters < embed_dim:
            warnings.warn(f"args.sf = {start_filters} < embed_dim used [{embed_dim}]")
        if start_filters > embed_dim:
            embed_dim = int(embed_dim * np.ceil(start_filters / embed_dim))

        super().__init__(depths=tuple([depth] * stages),
                         num_heads=tuple([heads_] * stages),
                         out_chans=n_classes,
                         in_chans=in_channels,
                         embed_dim=embed_dim,
                         img_size=(height, width),
                         # ape=True,
                         n_divs=n_divs,
                         patch_size=patch_size
                         )


class HyperHybridSwinTransformer(nn.Module):
    def __init__(self, in_channels, n_classes, start_filters, stages=3, depth=3,
                 # use_group_norm=False, blk_type='resnet', classifier='sigmoid',
                 net_type='sedenion', n_divs=None, activation='relu', inplace_activation=False, height=495, width=436,
                 memory_efficient=False, patch_size=1, head=8,
                 **kwargs):
        super().__init__()

        # Details here
        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        self.n_divs = n_divs

        in_planes = int(n_divs * np.ceil(start_filters / n_divs))
        if start_filters < in_planes:
            warnings.warn(f"start_filters = {start_filters} < in_planes used [{in_planes}]")

        num_blocks, act_layer = get_extras(depth, stages, activation)
        self.act_layer = act_layer

        self.encoder = Encoder(num_blocks, in_channels, in_planes, n_divs, act_layer, inplace_activation)

        feature_sizes = self.encoder.feature_sizes
        in_channels = feature_sizes[0]

        # code
        num_heads = (head, )
        net = SwinTransformer(img_size=(height, width),
                              use_checkpoint=memory_efficient,
                              patch_size=patch_size,
                              depths=(depth, ),
                              num_heads=num_heads,
                              out_indices=(0, ),
                              embed_dim=int(n_divs * max(num_heads) * np.ceil(in_channels /
                                                                              (n_divs * max(num_heads)))),
                              in_chans=in_channels,
                              n_divs=n_divs,
                              mlp_ratio=4)
        self.center = net

        self.decoders = nn.ModuleList()
        x_planes = in_channels
        for idx, y_planes in enumerate(feature_sizes):
            decoder = DecodeBlock(x_planes, y_planes, y_planes, n_divs, act_layer, scale_factor=2 if idx != 0 else 1)
            x_planes = y_planes
            self.decoders.append(decoder)

        self.head = nn.Sequential(
            nn.Conv2d(x_planes, n_classes, kernel_size=3, stride=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)

        x = self.act_layer(self.center(features[0]))

        for decoder, y in zip(self.decoders, features):
            x = decoder(x, y)


class HyperUNet(HyperTransformerUNet):
    def __init__(self,  # args,
                 n_channels, n_frame_out, start_filters, stages=3, depth=3, decode_depth=None, n_frame_in=12,
                 # use_group_norm=False, blk_type='resnet', classifier='sigmoid',
                 height=495, width=436, merge_type='concat', constant_dim=False,
                 memory_efficient=False, patch_size=1, dropout=0.0, use_neck=False,
                 **kwargs):

        super().__init__(depths=tuple([depth] * stages), #2,2,6,2
                         out_channels=n_frame_out * n_channels,
                         in_channels=n_frame_in * n_channels,
                         decode_depth=decode_depth,
                         embed_dim=start_filters,
                         img_size=(height, width),
                         patch_size=patch_size,
                         merge_type=merge_type,
                         drop_rate=dropout,
                         use_checkpoint=memory_efficient,
                         use_neck=use_neck,
                         with_sigmoid=True,  # False,
                         constant_dim=constant_dim,
                         )
                         

class HyperSwinEncoderDecoder3D(SwinEncoderDecoderTransformer3D):
    def __init__(self,  # args,
                 n_channels, n_frame_out, start_filters, stages=3, depth=3, decode_depth=None, n_frame_in=12,
                 # use_group_norm=False, blk_type='resnet', classifier='sigmoid',
                 net_type='sedenion', n_divs=None, height=495, width=436,
                 memory_efficient=False, patch_size=1, dropout=0.0, use_neck=False,
                 **kwargs):

        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(n_frame_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if start_filters < embed_dim:
            warnings.warn(f"start_filters = {start_filters} < embed_dim used [{embed_dim}]")
        if start_filters > embed_dim:
            embed_dim = int(embed_dim * np.ceil(start_filters / embed_dim))

        super().__init__(depths=tuple([depth] * stages),
                         num_heads=tuple([heads_] * stages),
                         decode_depth=decode_depth,
                         out_chans=n_frame_out,
                         in_chans=n_frame_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(height, width),
                         in_depth=n_channels, out_depth=n_channels,
                         n_divs=n_divs,
                         patch_size=(1, *([patch_size] * 2)),
                         use_checkpoint=memory_efficient,
                         drop_rate=dropout,
                         use_neck=use_neck,
                         )


class HyperSwinUNet3D(SwinUNet3D):
    def __init__(self,  # args
                 n_channels, n_frame_out, start_filters, stages=3, depth=3, decode_depth=None, n_frame_in=12,
                 merge_type='concat', mlp_ratio=4,
                 net_type='sedenion', n_divs=None, height=495, width=436,
                 memory_efficient=False, patch_size=1, dropout=0.0, use_neck=False,
                 mix_features=False, constant_dim=False,
                 **kwargs
                 ):
        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(n_frame_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if start_filters < embed_dim:
            warnings.warn(f"start_filters = {start_filters} < embed_dim used [{embed_dim}]")
        if start_filters > embed_dim:
            embed_dim = int(embed_dim * np.ceil(start_filters / embed_dim))

        if decode_depth:
            decode_depth = min(decode_depth, depth)  # decoding depth must not be more than encodeing depth

        super().__init__(depths=tuple([depth] * stages),
                         decode_depth=decode_depth,
                         num_heads=tuple([heads_] * stages),
                         out_chans=n_frame_out,
                         in_chans=n_frame_in,
                         merge_type=merge_type,
                         mlp_ratio=mlp_ratio,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(height, width),
                         in_depth=n_channels, out_depth=n_channels,
                         n_divs=n_divs,
                         patch_size=(1, *([patch_size] * 2)),
                         use_checkpoint=memory_efficient,
                         use_neck=use_neck,
                         drop_rate=dropout,
                         constant_dim=constant_dim,
                         window_size=(1, 8, 8),  # updated on sept 9 2021
                         mix_features=mix_features,
                         )


class HyperSwinUPerNet3D(SwinUPerNet3D):
    def __init__(self,  # args
                 n_channels, n_frame_out, start_filters, stages=3, depth=3, n_frame_in=12,
                 merge_type='concat',
                 net_type='sedenion', n_divs=None, height=495, width=436,
                 memory_efficient=False, patch_size=1, dropout=0.0,
                 **kwargs
                 ):
        if n_divs is None:
            n_divs = n_div_dict[net_type.lower()]

        heads_ = 8
        n_multiples_in = int(n_divs * np.ceil(n_frame_in / n_divs))
        embed_dim = int(n_divs * heads_ * np.ceil(n_multiples_in / (n_divs * heads_)))
        if start_filters < embed_dim:
            warnings.warn(f"start_filters = {start_filters} < embed_dim used [{embed_dim}]")
        if start_filters > embed_dim:
            embed_dim = int(embed_dim * np.ceil(start_filters / embed_dim))

        super().__init__(depths=tuple([depth] * stages),
                         num_heads=tuple([heads_] * stages),
                         out_chans=n_frame_out,
                         in_chans=n_frame_in,
                         embed_dim=embed_dim,  # args.sf,
                         img_size=(height, width),
                         in_depth=n_channels, out_depth=n_channels,
                         n_divs=n_divs,
                         patch_size=(1, *([patch_size] * 2)),
                         use_checkpoint=memory_efficient,
                         drop_rate=dropout,
                         )

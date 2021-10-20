import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from hypercomplex import (ComplexConv2D, QuaternionConv2D, OctonionConv2D, SedanionConv2D, get_c,
                          ComplexTransposeConv2D, QuaternionTransposeConv2D, OctonionTransposeConv2D,
                          SedanionTransposeConv2D)
# from fast_hypercomplex import (Concatenate as Concat,
#                                ComplexConv2d, QuaternionConv2d, SedenionConv2d, OctonionConv2d,
#                                ComplexConvTranspose2d, QuaternionConvTranspose2d, OctonionConvTranspose2d,
#                                SedenionConvTranspose2d)


conv_dict = {16: SedanionConv2D,
             8: OctonionConv2D,
             4: QuaternionConv2D,
             2: ComplexConv2D,
             1: nn.Conv2d}
transpose_conv_dict = {16: SedanionTransposeConv2D,
                       8: OctonionTransposeConv2D,
                       4: QuaternionTransposeConv2D,
                       2: ComplexTransposeConv2D,
                       1: nn.ConvTranspose2d}
# conv_dict = {16: SedenionConv2d,
#              8: OctonionConv2d,
#              4: QuaternionConv2d,
#              2: ComplexConv2d,
#              1: nn.Conv2d}
# transpose_conv_dict = {16: SedenionConvTranspose2d,
#                        8: OctonionConvTranspose2d,
#                        4: QuaternionConvTranspose2d,
#                        2: ComplexConvTranspose2d,
#                        1: nn.ConvTranspose2d}

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


class Concat(nn.Module):
    def __init__(self, n_divs, dim=1):
        super().__init__()
        self.n_divs = n_divs
        self.dim = dim

    def forward(self, x):
        components = [torch.cat([get_c(x_i, component, self.n_divs) for x_i in x], dim=1) for
                      component in range(self.n_divs)]
        return torch.cat(components, dim=self.dim)


class Activation(nn.Sequential):
    def __init__(self, activation, inplace=False, modify_activation=False):
        super(Activation, self).__init__()
        act = activation_dict[activation]
        if hasattr(act, 'inplace'):
            act.inplace = True if inplace else act.inplace
        if hasattr(act, 'min_val'):
            act.min_val = 0 if modify_activation else act.min_val
        self.add_module('activation', act)


class GroupNormBlock(nn.Module):  # Sequential):
    def __init__(self, num_channels, eps=1e-6, n_divs=1):
        super().__init__()
        if n_divs == 1:  # net_type is real
            gn = nn.GroupNorm(num_groups=8, num_channels=num_channels, eps=eps)
        else:
            gn = nn.GroupNorm(num_groups=n_divs, num_channels=num_channels, eps=eps)
        self.num_features = num_channels
        setattr(self, 'gn', gn)

    def forward(self, x):
        return cp.checkpoint(self.gn, x)


class NormBlock(nn.Sequential):
    def __init__(self, num_channels, eps=1e-6, n_divs=1, use_group_norm=True):
        super(NormBlock, self).__init__()
        self.num_channels = num_channels
        if use_group_norm:
            norm = GroupNormBlock(num_channels=num_channels, eps=eps, n_divs=n_divs)
        else:
            norm = nn.BatchNorm2d(num_features=num_channels, eps=eps)
        self.add_module('norm', norm)


class ConvActNorm(nn.Sequential):
    def __init__(self, in_channels, h_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), n_divs=1,
                 drop_rate=0.0):
        super().__init__()
        # self.add_module('conv', nn.Conv2d(in_channels, h_size, kernel_size=kernel_size, stride=stride,
        #                                   padding=padding))
        # self.add_module('act', nn.ELU(inplace=True))
        # self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size, eps=1e-6))

        self.add_module('conv', conv_dict[n_divs](in_channels, h_size, kernel_size=kernel_size, stride=stride,
                                                  padding=padding))
        self.drop_rate = float(drop_rate)
        if self.drop_rate > 0:
            self.add_module('drop', nn.Dropout2d(p=self.drop_rate, inplace=True))
        self.add_module('act', nn.ELU(inplace=True))
        # self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size, eps=1e-6))
        self.add_module('norm', NormBlock(num_channels=h_size, eps=1e-6, n_divs=n_divs))


class Conv1x1ActNorm(nn.Sequential):
    def __init__(self, in_channels, h_size, n_divs=1, drop_rate=0.0):
        super().__init__()
        self.add_module('conv', ConvActNorm(in_channels, h_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                            n_divs=n_divs, drop_rate=drop_rate))
        # self.add_module('conv', nn.Conv2d(in_channels, h_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))
        # self.add_module('act', nn.ELU(inplace=True))
        # self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size, eps=1e-6))


class Conv3x3ActNorm(nn.Sequential):
    def __init__(self, in_channels, h_size, n_divs=1, drop_rate=0.0):
        super().__init__()
        self.add_module('conv', ConvActNorm(in_channels, h_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                            n_divs=n_divs, drop_rate=drop_rate))
        # self.add_module('conv', nn.Conv2d(in_channels, h_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.add_module('act', nn.ELU(inplace=True))
        # self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size, eps=1e-6))


class PoolConvActNorm(nn.Sequential):
    def __init__(self, in_channels, h_size, n_divs=1, drop_rate=0.0):
        super().__init__()
        self.add_module('pool', nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)))
        # self.add_module('conv', nn.Conv2d(in_channels, h_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        # self.add_module('act', nn.ELU(inplace=True))
        # self.add_module('norm', nn.GroupNorm(num_groups=8, num_channels=h_size, eps=1e-6))
        self.add_module('conv', Conv3x3ActNorm(in_channels=in_channels, h_size=h_size, n_divs=n_divs,
                                               drop_rate=drop_rate))


class ConcatOutput(nn.Module):
    def __init__(self, net_in, dim=1, n_divs=1):
        super().__init__()
        self.net_in = net_in
        # self.dim = dim
        self.concat = Concat(n_divs=n_divs, dim=dim)

    def forward(self, x):
        # x_out = [x, self.net_in(x)]
        # return torch.cat(x_out, dim=self.dim)
        return self.concat([x, self.net_in(x)])


class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=4):
        super().__init__()
        self._se_ratio = se_ratio
        # num_squeezed_channels = num_squeezed_channels or in_channels
        self.act_layer = nn.ELU(inplace=True)

        # Squeeze and Excitation layer
        num_squeezed_channels = max(1, in_channels // self._se_ratio)
        self._se_reduce = nn.Conv2d(in_channels=in_channels, out_channels=num_squeezed_channels, kernel_size=(1, 1))
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, x):
        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self.act_layer(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x
        return x


class UpscaleConcatConvActNorm(nn.Module):
    def __init__(self, x_channels, y_channels, h_size, n_divs=1, drop_rate=0.0, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.x_layer = nn.Sequential(
            transpose_conv_dict[n_divs](x_channels, h_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                        output_padding=(1, 1)),
            nn.ELU(inplace=True)
        )
        self.xy_layer = Conv3x3ActNorm(h_size + y_channels, h_size, n_divs=n_divs, drop_rate=drop_rate)
        self.concat = Concat(n_divs=n_divs, dim=1)
        # Squeeze and Excitation layer, if desired
        if self.use_se:
            self.se_block = SEBlock(in_channels=h_size)

    def forward(self, x, y):
        # xs_ = x.shape
        x = self.x_layer(x)
        hy, wy = y.shape[2:]
        x = x[:, :, :hy, :wy]

        # print(xs_, x.shape, y.shape)
        # xy = torch.cat([x, y], dim=1)
        xy = self.concat([x, y])

        xy = self.xy_layer(xy)
        if hasattr(self, 'se_block'):  # if using squeeze nd excitation
            xy = self.se_block(xy)

        return xy


class UpscaleConcat(nn.Module):
    def __init__(self, x_channels: list, y_channels: list, h_size, n_divs=1, drop_rate=0.0, use_se=False):
        super().__init__()
        in_channels = 0
        self.x_layers = nn.ModuleList()
        for i, x_channel in enumerate(x_channels):
            # stride = tuple([2 ** (i + 1)] * 2)
            scale_factor = 2 ** (i + 1)
            x_layer = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                Conv3x3ActNorm(x_channel, h_size, n_divs, drop_rate=drop_rate)
                # transpose_conv_dict[n_divs](x_channel, h_size, kernel_size=(3, 3), stride=stride,
                #                             padding=(1, 1), output_padding=(1, 1)),
                # nn.ELU(inplace=True)
            )
            in_channels += h_size
            self.x_layers.append(x_layer)

        # self.x_layer = nn.Sequential(
        #     transpose_conv_dict[n_divs](x_channels, h_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
        #                                 output_padding=(1, 1)),
        #     nn.ELU(inplace=True)
        # )
        self.y_layers = nn.ModuleList()
        self.y_layers.append(Conv3x3ActNorm(y_channels[0], h_size, n_divs, drop_rate=drop_rate))
        in_channels += h_size
        for i, y_channel in enumerate(y_channels[1:]):
            factor = tuple([2 ** (i + 1)] * 2)
            y_layer = nn.Sequential(nn.AvgPool2d(kernel_size=factor, stride=factor, ceil_mode=True),
                                    Conv3x3ActNorm(y_channel, h_size, n_divs, drop_rate=drop_rate))
            in_channels += h_size
            self.y_layers.append(y_layer)

        self.xy_layer = Conv3x3ActNorm(in_channels, h_size, n_divs=n_divs, drop_rate=drop_rate)
        self.concat = Concat(n_divs=n_divs, dim=1)
        # Squeeze and Excitation layer, if desired
        self.use_se = use_se
        if self.use_se:
            self.se_block = SEBlock(in_channels=h_size)

    def forward(self, x_list, y_list):
        hy, wy = y_list[0].shape[2:]

        features = []
        for layer, x in zip(self.x_layers, x_list):
            features.append(layer(x)[:, :, :hy, :wy])

        for layer, y in zip(self.y_layers, y_list):
            features.append(layer(y)[:, :, :hy, :wy])

        # print(xs_, x.shape, y.shape)
        # xy = torch.cat([x, y], dim=1)
        # print([k.shape for k in features])
        xy = self.concat(features)

        xy = self.xy_layer(xy)
        if hasattr(self, 'se_block'):  # if using squeeze nd excitation
            xy = self.se_block(xy)

        return xy

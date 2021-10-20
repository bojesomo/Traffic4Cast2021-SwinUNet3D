'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
modified: Alabi Bojesomo
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from fast_hypercomplex import (ComplexConv2d, QuaternionConv2d, OctonionConv2d, SedenionConv2d, Concatenate,
                               ComplexDropout2d, QuaternionDropout2d, OctonionDropout2d, SedenionDropout2d)

from torch.autograd import Variable


conv_dict = {16: SedenionConv2d,
             8: OctonionConv2d,
             4: QuaternionConv2d,
             2: ComplexConv2d,
             1: nn.Conv2d}


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, act_layer=F.relu, inplace_activation=False, se_ratio=4):
        super().__init__()
        self._se_ratio = se_ratio
        # num_squeezed_channels = num_squeezed_channels or in_channels
        self.act_layer = act_layer
        self.inplace = inplace_activation

        # Squeeze and Excitation layer
        num_squeezed_channels = max(1, in_channels // self._se_ratio)
        self._se_reduce = nn.Conv2d(in_channels=in_channels, out_channels=num_squeezed_channels, kernel_size=(1, 1))
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=in_channels, kernel_size=(1, 1))

    def forward(self, x):
        # Squeeze and Excitation
        x_squeezed = F.adaptive_avg_pool2d(x, 1)
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self.act_layer(x_squeezed, inplace=self.inplace)
        x_squeezed = self._se_expand(x_squeezed)
        x = torch.sigmoid(x_squeezed) * x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', n_divs=1, act_layer=F.relu, inplace_activation=False,
                 use_se=False):
        super(BasicBlock, self).__init__()

        self.use_se = use_se
        self.act_layer = act_layer
        self.inplace = inplace_activation

        # Squeeze and Excitation layer, if desired
        if self.use_se:
            self.se_block = SEBlock(in_channels=in_planes, act_layer=self.act_layer, inplace_activation=self.inplace)

        self.conv1 = conv_dict[n_divs](in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_dict[n_divs](planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     conv_dict[n_divs](in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        if hasattr(self, 'se_block'):  # if using squeeze nd excitation
            x = self.se_block(x)

        out = self.act_layer(self.bn1(self.conv1(x)), inplace=self.inplace)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        # out = F.relu(out)
        out = self.act_layer(out, inplace=self.inplace)
        return out


class LearnVector(nn.Module):
    def __init__(self, in_planes, planes, act_layer=F.relu, concatenate=True, inplace_activation=False):
        super().__init__()
        self.act_layer = act_layer
        self.inplace = inplace_activation
        self.concatenate = concatenate

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        out = self.conv1(self.act_layer(self.bn1(x), inplace=self.inplace))
        out = self.conv2(self.act_layer(self.bn2(out), inplace=self.inplace))
        if self.concatenate:
            out = torch.cat([x, out], dim=1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, in_planes=16,  # num_classes=10,
                 n_divs=1, act_layer=F.relu, inplace_activation=False,
                 option='A', concatenate=True, constant_dim=False, use_se=False,
                 ):
        super(ResNet, self).__init__()
        self.act_layer = act_layer
        self.inplace = inplace_activation
        self.in_channels = in_channels
        self.in_planes = in_planes  # 16
        self.n_divs = n_divs
        self.option = option
        self.use_se = use_se

        if n_divs > 1:
            n_channels = in_channels * (n_divs - 1) if concatenate else (n_divs * math.ceil(in_channels / n_divs))
            self.vectors = LearnVector(in_channels, n_channels, act_layer=act_layer,
                                       inplace_activation=inplace_activation,
                                       concatenate=concatenate)
            in_channels = in_channels * n_divs if concatenate else n_channels

        self.conv1 = conv_dict[n_divs](in_channels, in_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                       bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.feature_sizes = [in_planes]

        self.layers = nn.ModuleList()
        multiplier = {True: 1, False: 2}[constant_dim]
        for ind, num_block in enumerate(num_blocks):
            in_planes *= multiplier if ind > 0 else 1
            layer = self._make_layer(block, in_planes, num_block, stride=1 if ind == 0 else 2)
            self.layers.append(layer)
            self.feature_sizes.insert(0, in_planes)
        # self.linear = nn.Linear(in_planes, num_classes)

        if n_divs == 1:
            self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes, planes, stride, option=self.option, n_divs=self.n_divs, act_layer=self.act_layer,
                      inplace_activation=self.inplace, use_se=self.use_se))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.n_divs > 1:
            x = self.vectors(x)

        features = []
        out = self.act_layer(self.bn1(self.conv1(x)), inplace=self.inplace)
        features.insert(0, out)
        for layer in self.layers:
            out = layer(out)
            features.insert(0, out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return features  # out


class Encoder(ResNet):
    def __init__(self,  # block,
                 num_blocks, in_channels=3, in_planes=16,  # num_classes=10,
                 n_divs=1, act_layer=F.relu, inplace_activation=False, constant_dim=False, use_se=False,
                 # option='A',
                 **kwargs):
        block = BasicBlock
        option = 'B'
        concatenate = False
        self.kwargs = kwargs
        super().__init__(block, num_blocks, in_channels, in_planes,  # num_classes,
                         n_divs=n_divs, act_layer=act_layer, option=option, concatenate=concatenate,
                         inplace_activation=inplace_activation, constant_dim=constant_dim,
                         use_se=use_se)


class DecodeBlock(nn.Module):
    def __init__(self, x_planes, y_planes, planes, n_divs=1, act_layer=F.relu, scale_factor=2,
                 inplace_activation=False):
        super().__init__()
        self.act_layer = act_layer
        self.inplace = inplace_activation
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor) if scale_factor > 1 else nn.Identity(),
            conv_dict[n_divs](x_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.layer2 = nn.Sequential(
            conv_dict[n_divs](planes + y_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.concatenate = Concatenate(n_divs=n_divs, dim=1)

    def forward(self, x, y):
        y_shape = list(y.shape)
        hy, wy = y_shape[2:]

        x = self.act_layer(self.layer1(x), inplace=self.inplace)
        x = x[:, :, :hy, :wy]

        x = self.concatenate([x, y])

        x = self.act_layer(self.layer2(x), inplace=self.inplace)

        return x


class HyperUnet(nn.Module):
    def __init__(self, num_blocks, in_channels=3, in_planes=16,  num_classes=1,
                 n_divs=1, act_layer=F.relu, inplace_activation=False, constant_dim=False,
                 use_se=False,
                 **kwargs):
        super().__init__()
        self.act_layer = act_layer
        self.inplace = inplace_activation

        self.encoder = Encoder(num_blocks, in_channels, in_planes, n_divs, act_layer, inplace_activation,
                               constant_dim, use_se)

        feature_sizes = self.encoder.feature_sizes
        in_planes = feature_sizes[0]
        self.center = nn.Sequential(
            conv_dict[n_divs](in_planes, 2 * in_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_planes)
        )
        x_planes = 2 * in_planes
        self.decoders = nn.ModuleList()
        for idx, y_planes in enumerate(feature_sizes):
            decoder = DecodeBlock(x_planes, y_planes, y_planes, n_divs, act_layer, scale_factor=2 if idx != 0 else 1,
                                  inplace_activation=inplace_activation)
            x_planes = y_planes
            self.decoders.append(decoder)

        self.head = nn.Sequential(
            nn.Conv2d(x_planes, num_classes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)

        x = self.act_layer(self.center(features[0]), inplace=self.inplace)

        for decoder, y in zip(self.decoders, features):
            x = decoder(x, y)

        return self.head(x)

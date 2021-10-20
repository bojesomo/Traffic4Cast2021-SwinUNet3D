from .layers import *


class DenseBlock(nn.Module):
    def __init__(self, in_channels, h_size):
        super().__init__()
        hidden_channels = in_channels
        self.layers = nn.ModuleList()

        self.layers.append(ConcatOutput(nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                     ceil_mode=True)))
        hidden_channels += in_channels

        self.layers.append(ConcatOutput(Conv1x1ActNorm(hidden_channels, h_size)))
        hidden_channels += h_size

        self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size//2)))
        hidden_channels += h_size // 2

        self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size // 2)))
        hidden_channels += h_size // 2

        self.out_layer = Conv1x1ActNorm(hidden_channels, h_size)

    def forward(self, x):

        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        x = self.out_layer(x)
        return x


class ConvPoolDenseBlock(nn.Sequential):
    def __init__(self, in_channels, h_size):
        super().__init__()
        self.add_module('pool', ConvActNorm(in_channels, h_size, kernel_size=(3, 3), stride=(2, 2),
                                            padding=(1, 1)))
        # self.add_module('pool', nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True))
        self.add_module('dense', DenseBlock(in_channels, h_size))


class NetA(nn.Module):
    def __init__(self, input_channels, out_size):
        super().__init__()
        self.encode1 = DenseBlock(input_channels, 128)
        self.encode2 = ConvPoolDenseBlock(128, 128)
        self.encode3 = ConvPoolDenseBlock(128, 128)
        self.encode4 = ConvPoolDenseBlock(128, 128)
        self.encode5 = ConvPoolDenseBlock(128, 128)
        self.encode6 = ConvPoolDenseBlock(128, 128)
        self.encode7 = ConvPoolDenseBlock(128, 128)
        self.encode8 = ConvPoolDenseBlock(128, 128)

        self.neck = Conv3x3ActNorm(128, 128)

        self.decode7 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode6 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode5 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode4 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode3 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode2 = UpscaleConcatConvActNorm(128, 128, 128)
        self.decode1 = UpscaleConcatConvActNorm(128, 128, 128)

        self.head = nn.Sequential(
            nn.Conv2d(128, out_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, image_input):
        x1 = self.encode1(image_input)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)
        x6 = self.encode6(x5)
        x7 = self.encode7(x6)
        x8 = self.encode8(x7)

        x100 = self.neck(x8)

        x107 = self.decode7(x100, x7)
        x106 = self.decode6(x107, x6)
        x105 = self.decode5(x106, x5)
        x104 = self.decode4(x105, x4)
        x103 = self.decode3(x104, x3)
        x102 = self.decode2(x103, x2)
        x101 = self.decode1(x102, x1)

        out = self.head(x101)

        return out


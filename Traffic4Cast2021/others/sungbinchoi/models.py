from .layers import *
import numpy as np
from timm.models.layers import to_ntuple


class DenseBlockA(nn.Module):
    def __init__(self, in_channels, h_size, nb_layers, n_divs=1, drop_rate=0.0):
        super().__init__()
        hidden_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            layer = ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size, n_divs=n_divs, drop_rate=drop_rate),
                                 n_divs=n_divs)
            hidden_channels += h_size
            self.layers.append(layer)

        self.out_layer = Conv1x1ActNorm(hidden_channels, h_size, n_divs=n_divs, drop_rate=drop_rate)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class DenseBlockB(nn.Module):
    def __init__(self, in_channels, h_size, nb_layers=4, n_divs=1, drop_rate=0.0):
        super().__init__()
        assert nb_layers >= 2, f"nb_layers, {nb_layers} must be greater than 1"
        hidden_channels = in_channels
        self.layers = nn.ModuleList()

        self.layers.append(ConcatOutput(Conv1x1ActNorm(hidden_channels, h_size, n_divs=n_divs, drop_rate=drop_rate),
                                        n_divs=n_divs))
        hidden_channels += h_size

        # self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size//2)))
        # hidden_channels += h_size//2
        #
        # self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size // 2)))
        # hidden_channels += h_size // 2

        for _ in range(1, nb_layers - 1):
            self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size // 2, n_divs=n_divs,
                                                           drop_rate=drop_rate), n_divs=n_divs))
            hidden_channels += h_size // 2

        self.layers.append(ConcatOutput(nn.Sequential(Conv1x1ActNorm(hidden_channels, h_size // 2, n_divs=n_divs,
                                                                     drop_rate=drop_rate),
                                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                                   ceil_mode=True)
                                                      ), n_divs=n_divs
                                        )
                           )
        hidden_channels += h_size // 2

        self.out_layer = Conv1x1ActNorm(hidden_channels, h_size, n_divs, drop_rate=drop_rate)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        x = self.out_layer(x)
        return x


class DenseBlockC(nn.Module):
    def __init__(self, in_channels, h_size, nb_layers, n_divs=1, drop_rate=0.0):
        super().__init__()
        assert nb_layers >= 2, f"nb_layers, {nb_layers} must be greater than 1"
        hidden_channels = in_channels
        self.layers = nn.ModuleList()

        self.layers.append(ConcatOutput(nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                     ceil_mode=True), n_divs=n_divs))
        hidden_channels += in_channels

        self.layers.append(ConcatOutput(Conv1x1ActNorm(hidden_channels, h_size, n_divs=n_divs, drop_rate=drop_rate),
                                        n_divs=n_divs))
        hidden_channels += h_size

        # self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size//2)))
        # hidden_channels += h_size // 2
        #
        # self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size // 2)))
        # hidden_channels += h_size // 2

        for _ in range(2, nb_layers):
            self.layers.append(ConcatOutput(Conv3x3ActNorm(hidden_channels, h_size // 2, n_divs=n_divs,
                                                           drop_rate=drop_rate), n_divs=n_divs))
            hidden_channels += h_size // 2

        self.out_layer = Conv1x1ActNorm(hidden_channels, h_size, n_divs, drop_rate=drop_rate)

    def forward(self, x):

        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        x = self.out_layer(x)
        return x


class DenseBlockD(nn.Module):
    def __init__(self, in_channels, h_size, nb_layers, n_divs=1, drop_rate=0.0):
        super().__init__()
        local_h_size = int(np.ceil(h_size / (nb_layers * n_divs)) * n_divs)
        hidden_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(nb_layers):
            layer = ConcatOutput(Conv3x3ActNorm(hidden_channels, local_h_size, n_divs=n_divs, drop_rate=drop_rate),
                                 n_divs=n_divs)
            hidden_channels += local_h_size
            # print(layer)
            self.layers.append(layer)

        self.out_layer = Conv1x1ActNorm(hidden_channels, h_size, n_divs=n_divs, drop_rate=drop_rate)

    def forward(self, x):

        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        x = self.out_layer(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_channels, h_size, nb_layers=None, dense_type='A', n_divs=1, drop_rate=0.0, use_se=False):
        super().__init__()
        net = {'A': DenseBlockA,
               'B': DenseBlockB,
               'C': DenseBlockC,
               'D': DenseBlockD}[dense_type](in_channels, h_size, nb_layers, n_divs, drop_rate=drop_rate)
        # net = {'A': DenseBlockA(in_channels, h_size, nb_layers)}[dense_type]
        self.use_se = use_se

        self.layers = net.layers
        self.out_layer = net.out_layer
        # Squeeze and Excitation layer, if desired
        if self.use_se:
            self.se_block = SEBlock(in_channels=h_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.out_layer(x)
        if hasattr(self, 'se_block'):  # if using squeeze nd excitation
            x = self.se_block(x)
        return x


class PoolDenseBlock(nn.Sequential):
    def __init__(self, in_channels, h_size, nb_layers=None, dense_type='A', n_divs=1, drop_rate=0.0, use_se=False):
        super().__init__()
        if dense_type == 'A':
            pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        else:
            pool = ConvActNorm(in_channels, in_channels, kernel_size=(3, 3), stride=(2, 2),
                               padding=(1, 1), n_divs=n_divs, drop_rate=drop_rate)

        self.add_module('pool', pool)
        self.add_module('dense', DenseBlock(in_channels, h_size, nb_layers, dense_type=dense_type, n_divs=n_divs,
                                            drop_rate=drop_rate, use_se=use_se))


class Net(nn.Module):
    def __init__(self, input_channels, out_size, hidden_size=128, nb_layers=tuple([2]*8),
                 encode_dims=(64, 96, 128, 128, 128, 128, 128, 128), dense_type='A', n_divs=1,
                 drop_rate=0.0, use_se=False):
        """
            Unet Version: Original
        """
        super().__init__()

        self.use_se = use_se
        nb_layers = to_ntuple(len(encode_dims))(nb_layers)
        # ########
        # Encoder : Dowsampling
        # ########
        self.encode = nn.ModuleList()
        self.encode.append(DenseBlock(input_channels, encode_dims[0], nb_layers[0], dense_type, n_divs,
                                      drop_rate=drop_rate, use_se=self.use_se))
        in_dim = encode_dims[0]
        for dim, nb_layer in zip(encode_dims[1:], nb_layers[1:]):
            self.encode.append(PoolDenseBlock(in_dim, dim, nb_layer, dense_type, n_divs, drop_rate=drop_rate,
                                              use_se=self.use_se))
            in_dim = dim

        self.neck = Conv3x3ActNorm(encode_dims[-1], hidden_size, n_divs=n_divs, drop_rate=drop_rate)
        # ########
        # Decoder : Upsampling
        # ##########
        dim_x = hidden_size
        self.decode = nn.ModuleList()
        for dim_y in encode_dims[::-1][1:]:
            self.decode.append(UpscaleConcatConvActNorm(dim_x, dim_y, hidden_size, n_divs=n_divs, drop_rate=drop_rate,
                                                        use_se=self.use_se))

        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encode = []
        for layer in self.encode:
            x = layer(x)
            x_encode.append(x)

        x = self.neck(x)

        for layer, y in zip(self.decode, x_encode[::-1][1:]):
            x = layer(x, y)

        out = self.head(x)
        return out


class Net3p(nn.Module):
    def __init__(self, input_channels, out_size, hidden_size=128, nb_layers=tuple([2]*8),
                 encode_dims=(64, 96, 128, 128, 128, 128, 128, 128), dense_type='A', n_divs=1,
                 drop_rate=0.0):
        """
            Unet3+ Version:
        """
        super().__init__()

        nb_layers = to_ntuple(len(encode_dims))(nb_layers)
        # ########
        # Encoder : Dowsampling
        # ########
        self.encode = nn.ModuleList()
        self.encode.append(DenseBlock(input_channels, encode_dims[0], nb_layers[0], dense_type, n_divs,
                                      drop_rate=drop_rate))
        in_dim = encode_dims[0]
        for dim, nb_layer in zip(encode_dims[1:], nb_layers[1:]):
            self.encode.append(PoolDenseBlock(in_dim, dim, nb_layer, dense_type, n_divs, drop_rate=drop_rate))
            in_dim = dim

        self.neck = Conv3x3ActNorm(encode_dims[-1], hidden_size, n_divs=n_divs, drop_rate=drop_rate)
        # ########
        # Decoder : Upsampling
        # ##########
        # dim_x = hidden_size
        self.decode = nn.ModuleList()
        decode_dims = encode_dims[::-1][1:]
        for i in range(len(decode_dims)):  # , dim_y in enumerate(decode_dims):
            dims_x = [hidden_size] * (i + 1)
            dims_y = decode_dims[i:]
            self.decode.append(UpscaleConcat(dims_x, dims_y, hidden_size, n_divs=n_divs, drop_rate=drop_rate))

        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, out_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_encode = []
        for layer in self.encode:
            # print(x.shape)
            x = layer(x)
            x_encode.append(x)

        # x = self.neck(x)

        # for layer, y in zip(self.decode, x_encode[::-1][1:]):
        #     x = layer(x, y)

        x_decode = [self.neck(x)]
        for i, layer in enumerate(self.decode):
            x = layer(x_decode, x_encode[::-1][i+1:])
            x_decode.insert(0, x)

        out = self.head(x_decode[0])
        return out


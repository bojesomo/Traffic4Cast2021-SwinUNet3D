from hypercomplex import get_comp_mat, get_c
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import to_ntuple
from .parametrize import register_parametrization


def get_comp(n_divs=4, comp_mat=None):
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    # comps = []
    # for i in range(n_divs):
    #     comp = comp_mat[np.abs(comp_mat) == i]
    #     comps.append(comp)
    abs_comp = np.abs(comp_mat)
    scale = np.sign(comp_mat) + np.eye(n_divs, n_divs)
    cc = [scale * (abs_comp == i) for i in range(n_divs)]
    return cc


def get_part(n_divs=4, comp_mat=None):
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing
    comp_mat[0, :] *= -1
    abs_comp = np.abs(comp_mat)
    scale = np.sign(comp_mat) + np.eye(n_divs, n_divs)
    cc = [scale * (abs_comp == i) for i in range(n_divs)]
    return cc


"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""
def kronecker_product(a, b):
    """
    Kronecker product of matrices a and b with leading batch dimensions.
    Batch dimensions are broadcast. The number of them mush
    :type a: torch.Tensor
    :type b: torch.Tensor
    :rtype: torch.Tensor
    """
    siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
    res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
    siz0 = res.shape[:-4]
    out = res.reshape(siz0 + siz1)
    return out


def phm(weights, n_divs=4, comp_mat=None):
    A = get_comp(n_divs, comp_mat)
    return sum([kronecker_product(weights[i], torch.from_numpy(A[i]).type_as(weights)) for i in range(n_divs)])


def make_hypercomplex_mul(weights, n_divs=4, split_dim=1, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    # return phm(weights, n_divs=n_divs, comp_mat=comp_mat)
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    # abs_comp = np.abs(comp_mat)
    # sign_comp = np.sign(comp_mat) + np.eye(n_divs)
    #
    # cat_kernels_hypercomplex = torch.cat([torch.cat([sign * weights[itr] for itr, sign in zip(abs_i, sign_i)], dim=0)
    #                                      for abs_i, sign_i in zip(abs_comp, sign_comp)], dim=1)
    def sign(ii):
        return np.sign(ii) if np.sign(ii) != 0 else 1

    neg_weights = - weights
    cat_kernels_hypercomplex = torch.cat([torch.cat([weights[np.abs(ii)] if sign(ii) > 0 else neg_weights[np.abs(ii)] for ii in comp_i], dim=0)
                                          for comp_i in comp_mat], dim=1)
    # cat_kernels_hypercomplex = torch.cat([torch.cat([sign(ii) * weights[np.abs(ii)] for ii in comp_i], dim=0)
    #                                       for comp_i in comp_mat], dim=1)

    # cat_kernel_hypercomplex_i = []
    # for comp_i in comp_mat:
    #     kernel_hypercomplex_i = []
    #     for idx, ii in enumerate(comp_i):
    #         itr = np.abs(ii)
    #         sign = np.sign(ii) if np.sign(ii) != 0 else 1
    #         kernel_hypercomplex_i.append(sign * weights[itr])
    #     cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=0))
    # cat_kernels_hypercomplex = torch.cat(cat_kernel_hypercomplex_i, dim=1)

    return cat_kernels_hypercomplex


def fast_hypercomplex(weights, n_divs=4, split_dim=1, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    # return make_hypercomplex_mul(weights, n_divs=n_divs, comp_mat=comp_mat)
    # return phm(weights, n_divs=n_divs, comp_mat=comp_mat)
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    # pos_comp = np.abs(comp_mat)
    # neg_comp = (comp_mat - pos_comp) // (-2)

    weights_new = torch.cat([weights, -torch.flipud(weights[1:])], dim=0)
    kernel = torch.cat([weights_new[comp_i].flatten(0, 1) for comp_i in comp_mat], dim=1)
    # weights_new = torch.zeros_like(weights)
    # weights_new[1:] = -2 * weights[1:]

    # pos_kernel = torch.cat([weights[comp_i].flatten(0, 1) for comp_i in pos_comp], dim=1)
    # neg_kernel = torch.cat([weights_new[comp_i].flatten(0, 1) for comp_i in neg_comp], dim=1)
    # pos_kernel = torch.stack([weights[comp_i].flatten(0, 1) for comp_i in pos_comp], dim=1).flatten(1,2)

    # zero_weight = torch.zeros_like(weights[0])
    # pos_kernel = torch.cat([torch.cat([weights[itr] for itr in comp_i], dim=0) for comp_i in pos_comp], dim=1)
    # neg_kernel = torch.cat([torch.cat([weights[itr] if itr != 0 else zero_weight for itr in comp_i], dim=0) for comp_i in neg_comp], dim=1)

    # return pos_kernel + neg_kernel
    # return pos_kernel - 2*neg_kernel
    # return pos_kernel
    return kernel


def multiply(q, v, n_divs, q_dim=-1, v_dim=-2):
    qs = torch.chunk(q, n_divs, dim=q_dim)

    vs = torch.chunk(v, n_divs, dim=v_dim)

    comp_mat = get_comp_mat(n_divs)

    cat_qv_i = []
    for comp_i in comp_mat:
        temp_qv_i = 0
        for idx, ii in enumerate(comp_i):
            itr = np.abs(ii)
            sign = np.sign(ii)
            temp_qv_i = temp_qv_i + (sign * torch.matmul(qs[itr], vs[itr]))
        cat_qv_i.append(temp_qv_i.unsqueeze(0))
    qv = torch.cat(cat_qv_i, dim=0)
    return qv
    # return torch.matrix_power(hamilton, B)


def product(q, v, n_divs, dim=-1):
    # q = q.transpose(dim, 1)
    # v = v.transpose(dim, 1)

    qs = torch.chunk(q, chunks=n_divs, dim=dim)
    vs = torch.chunk(v, chunks=n_divs, dim=dim)

    comp_mat = get_comp_mat(n_divs)

    cat_qv_i = []
    for comp_i in comp_mat:
        temp_qv_i = 0
        for idx, ii in enumerate(comp_i):
            itr = np.abs(ii)
            sign = np.sign(ii) if np.sign(ii) != 0 else 1
            temp_qv_i = temp_qv_i + (sign * qs[itr] * vs[itr])
            # temp_qv_i = temp_qv_i + (sign * torch.matmul(qs[itr], vs[itr]))
        cat_qv_i.append(temp_qv_i)  # .unsqueeze(0))
    qv = torch.cat(cat_qv_i, dim=dim)

    return qv


def dot_product(q, v, n_divs, dim=-1):
    qs = torch.chunk(q, chunks=n_divs, dim=dim)
    vs = torch.chunk(v, chunks=n_divs, dim=dim)

    qv = 0
    for i in range(n_divs):
        # sign = -1 if i > 0 else 1  # k*k = -1 for all imaginary components
        # qv += sign * torch.matmul(qs[i], vs[i].transpose(-2, -1))
        qv += torch.matmul(qs[i], vs[i].transpose(-2, -1))  # TODO -

    return qv


def component_product(q, v, n_divs, q_dim=-1, v_dim=-1):  # q, v, n_divs, q_dim=-1, v_dim=-2
    qs = torch.chunk(q, chunks=n_divs, dim=q_dim)
    vs = torch.chunk(v, chunks=n_divs, dim=dim)

    qv = 0
    for i in range(n_divs):
        # sign = -1 if i > 0 else 1  # k*k = -1 for all imaginary components
        # qv += sign * torch.matmul(qs[i], vs[i].transpose(-2, -1))
        qv += torch.matmul(qs[i], vs[i].transpose(-2, -1))  # TODO -

    return qv


def hypercomplex_dot_product(q0, q1, n_divs, dim=-1):
    qs = torch.chunk(q, chunks=n_divs, dim=dim)
    vs = torch.chunk(v, chunks=n_divs, dim=dim)

    qv = 0
    for i in range(n_divs):
        # sign = -1 if i > 0 else 1  # k*k = -1 for all imaginary components
        # qv += sign * torch.matmul(qs[i], vs[i].transpose(-2, -1))
        qv += torch.matmul(qs[i], vs[i].transpose(-2, -1))  # TODO -

    return qv


class Hamilton(nn.Module):
    def __init__(self, n_divs=4, comp_mat=None):
        super().__init__()
        self.register_buffer("n_divs", torch.tensor(n_divs))
        if comp_mat is None:
            comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing
        self.register_buffer('comp_mat', torch.from_numpy(comp_mat))

    def forward(self, weights):
        cat_kernel_hypercomplex_i = []
        for comp_i in self.comp_mat:
            kernel_hypercomplex_i = []
            for idx, ii in enumerate(comp_i):
                itr = np.abs(ii)
                sign = np.sign(ii) if np.sign(ii) != 0 else 1
                kernel_hypercomplex_i.append(sign * weights[itr])
            cat_kernel_hypercomplex_i.append(torch.cat(kernel_hypercomplex_i, dim=0))

        return torch.cat(cat_kernel_hypercomplex_i, dim=1)


# ############################
# '''Hypercomplex Linear!'''
# ###########################
class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, n_divs=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_divs = n_divs
        # self.weight = nn.Parameter(torch.FloatTensor(self.in_features // n_divs, self.out_features))
        self.weight = nn.Parameter(torch.FloatTensor(n_divs, self.out_features // n_divs, self.in_features // n_divs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.comp_mat = get_comp_mat(n_divs)
        self.reset_parameters()
        # changes
        # self.hamilton = make_hypercomplex_mul(self.weight, self.n_divs, comp_mat=self.comp_mat)
        # self.old_device = self.weight.device

    def reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        stdv = math.sqrt(6.0 / (self.weight.size(1) + self.weight.size(2) * self.n_divs))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    # def cuda(self, device=None):
    #     super().cuda(device)
    #     if self.weight.device != self.hamilton.device:
    #         # self.old_device = self.weight.device
    #         self.hamilton = self.hamilton.to(self.weight.device)
    #
    # def to(self, device=..., dtype=..., non_blocking=...):
    #     super().to(device, dtype, non_blocking)
    #     if self.weight.device != self.hamilton.device:
    #         # self.old_device = self.weight.device
    #         self.hamilton = self.hamilton.to(self.weight.device)

    def forward(self, x):
        # self.hamilton = self.hamilton.to(self.weight.device)
        self.hamilton = fast_hypercomplex(self.weight, self.n_divs, comp_mat=self.comp_mat)
        # if self.weight.device != self.hamilton.device:
        # #     self.old_device = self.weight.device
        # #     self.hamilton = self.hamilton.to(self.weight.device)
        #     self.hamilton = make_hypercomplex_mul(self.weight, self.n_divs, comp_mat=self.comp_mat)
        # else:
        #     self.hamilton = self.hamilton
        # self.hamilton = make_hypercomplex_mul(self.weight, self.n_divs, comp_mat=self.comp_mat)
        # output = torch.matmul(x, hamilton)
        # if self.bias is not None:
        #     output += self.bias
        # hamilton = self.hamilton.type(self.device)
        # print(self.hamilton.device)
        return F.linear(x, self.hamilton, self.bias)  # + F.linear(x, self.hamilton[1], None)
        # return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " + \
               f"bias={self.bias is not None}, n_divs={self.n_divs}"


class HyperLinear_(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, n_divs=4):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.n_divs = n_divs
        # self.weight = nn.Parameter(torch.FloatTensor(self.in_features // n_divs, self.out_features))
        self.weight = nn.Parameter(torch.FloatTensor(n_divs, self.out_features // n_divs, self.in_features // n_divs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.comp_mat = get_comp_mat(n_divs)
        self._reset_parameters()
        register_parametrization(self, 'weight', Hamilton(n_divs=self.n_divs, comp_mat=self.comp_mat))

    def _reset_parameters(self):
        # stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        stdv = math.sqrt(6.0 / (self.weight.size(1) + self.weight.size(2) * self.n_divs))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " + \
               f"bias={self.bias is not None}, n_divs={self.n_divs}"


# ############################
# '''Hypercomplex Conv'''
# ############################
class HyperConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='hypercomplex', operation='convolution2d',):
                 # hypercomplex_format=True):

        super(HyperConv, self).__init__()
        n = int(''.join(c for c in operation if c.isdigit()))
        self.n_divs = n_divs
        self.in_channels = in_channels   # // self.n_divs
        self.out_channels = out_channels  # // self.n_divs
        self.stride = to_ntuple(n)(stride)
        self.padding = to_ntuple(n)(padding)
        self.groups = groups
        self.dilation = to_ntuple(n)(dilation)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.operation = operation

        # self.hypercomplex_format = hypercomplex_format
        # self.winit = {'hypercomplex': hypercomplex_init,
        #               'unitary': unitary_init,
        #               'random': random_init}[self.weight_init]

        assert (self.in_channels % self.groups == 0)
        assert (self.out_channels % self.groups == 0)

        self.kernel_size, self.w_shape = self.get_kernel_and_weight_shape(kernel_size)

        self.weight = nn.Parameter(torch.Tensor(*self.w_shape))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.comp_mat = get_comp_mat(n_divs)
        self.reset_parameters()
        # changes
        # self.old_device = self.weight.device
        # self.cat_kernels_hypercomplex = None  # make_hypercomplex_mul(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)
        # self.cat_kernels_hypercomplex = make_hypercomplex_mul(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)

    def get_kernel_and_weight_shape(self, kernel_size):
        if self.operation == 'convolution1d':
            if type(kernel_size) is not int:
                raise ValueError(
                    """An invalid kernel_size was supplied for a 1d convolution. The kernel size
                    must be integer in the case. Found kernel_size = """ + str(kernel_size)
                )
            else:
                # ks = kernel_size
                ks = tuple((kernel_size,))
                # w_shape = (self.out_channels // self.n_divs, self.in_channels // self.groups) + (*ks,)
        else:  # in case it is 2d or 3d.
            if self.operation == 'convolution2d' and type(kernel_size) is int:
                ks = to_ntuple(2)(kernel_size)  # (kernel_size, kernel_size)
            elif self.operation == 'convolution3d' and type(kernel_size) is int:
                ks = to_ntuple(3)(kernel_size)  # (kernel_size, kernel_size, kernel_size)
            elif type(kernel_size) is not int:
                if self.operation == 'convolution2d' and len(kernel_size) != 2:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 2d convolution. The kernel size
                        must be either an integer or a tuple of 2. Found kernel_size = """ + str(kernel_size)
                    )
                elif self.operation == 'convolution3d' and len(kernel_size) != 3:
                    raise ValueError(
                        """An invalid kernel_size was supplied for a 3d convolution. The kernel size
                        must be either an integer or a tuple of 3. Found kernel_size = """ + str(kernel_size)
                    )
                else:
                    ks = kernel_size
            # w_shape = (out_channels, in_channels // self.groups) + (*ks,)
        w_shape = (self.n_divs, self.out_channels // self.n_divs, self.in_channels // self.groups // self.n_divs) +\
                  (*ks,)
        return ks, w_shape

    def reset_parameters(self):
        # print(self.w_shape)
        receptive_field = np.prod(self.w_shape[2:])
        fan_in = self.n_divs * self.w_shape[1] * receptive_field
        fan_out = self.w_shape[0] * receptive_field

        if self.init_criterion == 'glorot':
            stdv = np.sqrt(2 / (fan_in + fan_out))
        elif self.init_criterion == 'he':
            stdv = np.sqrt(2 / fan_in)
        else:
            raise ValueError('Invalid criterion: ' + self.init_criterion)
        # print(stdv)
        if self.weight_init == 'hypercomplex':
            stdv /= np.sqrt(self.n_divs)
        # print(stdv)
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.zero_()


    # def cuda(self, device=None):
    #     super().cuda(device)
    #     if self.weight.device != self.cat_kernels_hypercomplex.device:
    #         # self.old_device = self.weight.device
    #         self.cat_kernels_hypercomplex = self.cat_kernels_hypercomplex.to(self.weight.device)
    #
    # def to(self, device=..., dtype=..., non_blocking=...):
    #     super().to(device, dtype, non_blocking)
    #     if self.weight.device != self.cat_kernels_hypercomplex.device:
    #         # self.old_device = self.weight.device
    #         self.cat_kernels_hypercomplex = self.cat_kernels_hypercomplex.to(self.weight.device)

    def forward(self, x):
        assert 3 <= x.dim() <= 5, "The convolutional input x is either 3, 4 or 5 dimensional. x.dim = " + str(
            x.dim())
        # print(self.old_device, self.weight.device)
        # self.cat_kernels_hypercomplex = self.cat_kernels_hypercomplex.to(self.weight.device)
        self.cat_kernels_hypercomplex = fast_hypercomplex(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)
        # if self.weight.device != self.cat_kernels_hypercomplex.device:
        #     # self.old_device = self.weight.device
        #     # self.cat_kernels_hypercomplex = self.cat_kernels_hypercomplex.to(self.weight.device)
        #     self.cat_kernels_hypercomplex = make_hypercomplex_mul(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)
        # else:
        #     self.cat_kernels_hypercomplex = self.cat_kernels_hypercomplex
        # self.cat_kernels_hypercomplex = make_hypercomplex_mul(self.weight, n_divs=self.n_divs, comp_mat=self.comp_mat)  # updated 28May2021 for speed
        convfunc = {3: F.conv1d, 4: F.conv2d, 5: F.conv3d}[x.dim()]
        return convfunc(x, self.cat_kernels_hypercomplex, self.bias, self.stride, self.padding, self.dilation,
                        self.groups)  # + convfunc(x, self.cat_kernels_hypercomplex[1], None, self.stride, self.padding,
                                                # self.dilation, self.groups)

    def extra_repr(self) -> str:
        return 'in_channels=' + str(self.in_channels) \
               + ', out_channels=' + str(self.out_channels) \
               + ', bias=' + str(self.bias is not None) \
               + ', kernel_size=' + str(self.kernel_size) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', init_criterion=' + str(self.init_criterion) \
               + ', weight_init=' + str(self.weight_init) \
               + ', n_divs=' + str(self.n_divs) \
               + ', operation=' + str(self.operation)


class HyperConv1d(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='hypercomplex'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, n_divs=n_divs,
                         dilation=dilation, padding=padding, groups=groups, bias=bias, init_criterion=init_criterion,
                         weight_init=weight_init, operation='convolution1d')

    def __repr__(self):
            config = super().__repr__()
            config = config.replace(f', operation={str(self.operation)}', '')
            return config


class HyperConv2d(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='hypercomplex'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, n_divs=n_divs,
                         dilation=dilation, padding=padding, groups=groups, bias=bias, init_criterion=init_criterion,
                         weight_init=weight_init, operation='convolution2d')

    def __repr__(self):
            config = super().__repr__()
            config = config.replace(f', operation={str(self.operation)}', '')
            return config


class HyperConv3d(HyperConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, n_divs=4,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='hypercomplex'):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, n_divs=n_divs,
                         dilation=dilation, padding=padding, groups=groups, bias=bias, init_criterion=init_criterion,
                         weight_init=weight_init, operation='convolution3d')

    def __repr__(self):
            config = super().__repr__()
            config = config.replace(f', operation={str(self.operation)}', '')
            return config


class Concatenate(nn.Module):
    def __init__(self, dim=-1, n_divs=1):
        super(Concatenate, self).__init__()
        self.n_divs = n_divs
        self.dim = dim

    def forward(self, x):
        # dim = x[0].size(self.dim) // self.n_divs
        # x_splits = [torch.split(x_i, [dim for _ in range(self.n_divs)], dim=self.dim) for x_i in x]
        x_splits = [torch.chunk(x_i, chunks=self.n_divs, dim=self.dim) for x_i in x]
        components = [torch.cat([x_split[component] for x_split in x_splits], dim=self.dim) for component
                      in range(self.n_divs)]
        # components = [torch.cat([get_c(x_i, component, self.n_divs) for x_i in x], dim=self.dim) for component
        #               in range(self.n_divs)]
        return torch.cat(components, dim=self.dim)


class HyperSoftmax(nn.Module):
    def __init__(self, dim=-1, n_divs=4):
        super().__init__()
        self.softmax = nn.Softmax(dim=dim)
        self.n_divs = n_divs
        self.dim = dim
        
    def forward(self, x):
        # x_dim = list(x.shape)
        dim = x.size(self.dim) // self.n_divs
        x_split = torch.split(x, [dim for _ in range(self.n_divs)], dim=self.dim)
        components = [self.softmax(x_split[component]) for
                      component in range(self.n_divs)]
        return torch.cat(components, dim=self.dim)

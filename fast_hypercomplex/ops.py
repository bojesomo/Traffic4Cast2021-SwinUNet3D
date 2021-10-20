import numpy as np
import torch
from .utils import get_comp_mat


def get_comp(n_divs=4, comp_mat=None):
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

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


def make_hypercomplex_mul(weights, n_divs=4, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    def sign(ii):
        return np.sign(ii) if np.sign(ii) != 0 else 1

    neg_weights = - weights
    cat_kernels_hypercomplex = torch.cat(
        [torch.cat([weights[np.abs(ii)] if sign(ii) > 0 else neg_weights[np.abs(ii)] for ii in comp_i], dim=0)
         for comp_i in comp_mat], dim=1)
    return cat_kernels_hypercomplex


def fast_hypercomplex(weights, n_divs=4, comp_mat=None):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    if comp_mat is None:
        comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    weights_new = torch.cat([weights, -torch.flipud(weights[1:])], dim=0)
    kernel = torch.cat([weights_new[comp_i].flatten(0, 1) for comp_i in comp_mat], dim=1)
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


def hypercomplex_whitening(x, n_divs=4, dim=1, eps=1e-5):
    x_shape = x.shape
    reduction_dims = [t for t in range(x.dim()) if t != dim]
    broadcast_shape = [1 if t != dim else x_shape[dim] for t in range(x.dim())]
    mu = x.mean(reduction_dims).view(broadcast_shape)

    x -= mu
    x = torch.stack(torch.chunk(x, chunks=n_divs, dim=dim))
    # print(x.shape)
    # zero = torch.zeros(x.shape[1:])
    reduction_dims = [t + 1 for t in reduction_dims]
    broadcast_shape[dim] //= n_divs
    w = []
    for row in range(n_divs):
        b_ = list(range(row, n_divs))
        # print(b_)
        v_ = (x[b_] * x[row]).mean(reduction_dims)  # get the covariance (needed ones only)
        v_[0] += eps  # diagonal components
        # print(v_.shape)
        w_ = []
        for col in range(n_divs):
            if row > col:
                w_temp = w[col][row]
            elif row == col:
                extra = 0
                for k in range(row-1):
                    extra = extra + w[k][row] ** 2
                w_temp = torch.sqrt(v_[0].view(broadcast_shape) - extra).view(broadcast_shape)
            else:
                extra = 0
                for k in range(row-1):
                    extra = extra + w[k][row] * w[k][col]
                w_temp = (1 / w_[row]) * (v_[col - row].view(broadcast_shape) - extra)
            w_.append(w_temp)
        w.append(w_)
    s = sum([torch.stack(w_) * x_ for w_, x_ in zip(w, x)])
    return torch.cat([*s], dim=dim)


def get_modulus(x, n_divs=4, dim=1):
    # xs = torch.chunk(x, chunks=n_divs, dim=dim)
    left_params = [f'p{i + 1}' for i in range(x.ndim - 1)]
    right_params = [f'p{i + 1}' for i in range(x.ndim - 1)]
    left_params.insert(dim, '(n_divs c)')
    right_params.insert(dim, 'c')
    right_params.insert(0, 'n_divs')
    x = rearrange(x, f"{' '.join(left_params)} -> {' '.join(right_params)}", n_divs=n_divs)
    # x = rearrange(x, 'b (n_divs c) h w -> n_divs b c h w', n_divs=n_divs)
    x = (x ** 2).sum(0).sqrt_()
    return x


def get_normalized(x, eps=0.0001, n_divs=4, dim=1):
    left_params = [f'p{i + 1}' for i in range(x.ndim)]
    right_params = [f'p{i + 1}' for i in range(x.ndim)]
    right_params[dim] = f'(n_divs {right_params[dim]})'

    # broadcast_dims = [1] * x.ndim
    # broadcast_dims[dim] = n_divs
    x_modulus = get_modulus(x, n_divs=n_divs, dim=dim)
    # x = list(torch.chunk(x, chunks=n_divs, dim=dim))
    # x = [x_ / x_modulus for x_ in x]
    # x_modulus = x_modulus.repeat(*broadcast_dims)  # .expand_as(x)
    x_modulus = repeat(x_modulus, f"{' '.join(left_params)} -> {' '.join(right_params)}", n_divs=n_divs)
    return x / (x_modulus + eps)


def hypercomplex_exp(x, n_divs=4, dim=1):
    assert (x.shape[dim] % n_divs) == 0
    n = x.shape[dim] // n_divs
    a, x1 = torch.split(x, [n, n * (n_divs - 1)], dim=dim)
    x1_modulus = get_modulus(x1, n_divs=(n_divs - 1), dim=dim)

    x1s = list(torch.chunk(x1, chunks=(n_divs - 1), dim=dim))

    exp = torch.exp(a)
    exp_modulus = exp / x1_modulus
    for i, x_loc in enumerate(x1s):
        x1s[i] *= torch.sin(x1_modulus) * exp_modulus
    return torch.cat([exp * torch.cos(x1_modulus), *x1s], dim=dim)


def hypercomplex_log(x, n_divs=4, dim=1):
    assert (x.shape[dim] % n_divs) == 0
    x_modulus = get_modulus(x, n_divs=n_divs, dim=dim)
    n = x.shape[dim] // n_divs
    a, x1 = torch.split(x, [n, n * (n_divs - 1)], dim=dim)
    x1_normalized = get_normalized(x1, n_divs=(n_divs - 1), dim=dim)
    theta = torch.arccos(a / x_modulus)

    left_params = [f'p{i + 1}' for i in range(x1.ndim)]
    right_params = [f'p{i + 1}' for i in range(x1.ndim)]
    right_params[dim] = f'(n_divs {right_params[dim]})'
    theta = repeat(theta, f"{' '.join(left_params)} -> {' '.join(right_params)}", n_divs=(n_divs - 1))

    x1 = theta * x1_normalized

    return torch.cat([torch.log(x_modulus), x1], dim=dim)


def hypercomplex_sigmoid(x, n_divs=4, dim=1):
    assert n_divs > 1, f'n_divs={n_divs} but this function only works for n_divs>=2'
    x = hypercomplex_exp(x, n_divs=n_divs, dim=dim)
    x.narrow(dim=dim, start=0, length=x.shape[dim] // n_divs).add_(1)
    return x
    # n = x.shape[dim] // n_divs
    # a, x1 = torch.split(x, [n, n * (n_divs - 1)], dim=dim)
    # a += 1
    # return torch.cat([a, x1], )


def hypercomplex_tanh(x, n_divs=4, dim=1):
    assert n_divs > 1, f'n_divs={n_divs} but this function only works for n_divs>=2'
    x = 2 * hypercomplex_exp(x, n_divs=n_divs, dim=dim)
    x.narrow(dim=dim, start=0, length=x.shape[dim] // n_divs).add_(1)
    return x


def hypercomplex_softplus(x, n_divs=4, dim=1):
    assert n_divs > 1, f'n_divs={n_divs} but this function only works for n_divs>=2'
    x = hypercomplex_sigmoid(x, n_divs=n_divs, dim=dim)
    return hypercomplex_log(x)


def fast_product(x, y, n_divs=4, dim=1):
    """
    The constructed 'hamilton' W is a modified version of the hypercomplex representation,
    """
    comp_mat = get_comp_mat(n_divs)  # TODO - move this to the caller to reduce timing

    x = torch.stack(torch.chunk(x, chunks=n_divs, dim=dim))
    y = torch.stack(torch.chunk(y, chunks=n_divs, dim=dim))
    x = torch.cat([x, -torch.flipud(x[1:])], dim=0)
    # xy = torch.cat([x[comp_i].flatten(0, 1) * y for comp_i in comp_mat], dim=dim)
    xy = torch.cat([(x[comp_i] * y).sum(0) for comp_i in comp_mat], dim=dim)
    return xy


def hypercomplex_silu(x, n_divs=4, dim=0):
    sigmoid = hypercomplex_sigmoid(x, n_divs=n_divs, dim=dim)
    return fast_product(x, sigmoid, n_divs=n_divs, dim=dim)


def hypercomplex_mish(x, n_divs=4, dim=0):
    assert n_divs > 1, f'n_divs={n_divs} but this function only works for n_divs>=2'
    y = hypercomplex_exp(x, n_divs=n_divs, dim=dim)
    y.narrow(dim=dim, start=0, length=y.shape[dim] // n_divs).add_(2)
    return fast_product(x, y, n_divs=n_divs, dim=dim)
    #
    # x_softplus = hypercomplex_softplus(x, n_divs=n_divs, dim=dim)
    # x_tanh = hypercomplex_tanh(x_softplus, n_divs=n_divs, dim=dim)
    # return fast_product(x, x_tanh, n_divs=n_divs, dim=dim)

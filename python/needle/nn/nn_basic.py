"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device = device, dtype = dtype).reshape((1, out_features)), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = ops.matmul(X, self.weight)
        if self.bias:
            y += self.bias.broadcast_to(y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if X is None:
            return X
        for i, w in enumerate(X.shape):
            if i == 0:
                sum = 1
            else:
                sum = sum * w
        return X.reshape((X.shape[0], sum))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = x
        for module in self.modules:
            y = module(y)
        return y
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        log_sum_exp = ops.logsumexp(logits, axes = (1,))
        y_one_hot = init.one_hot(logits.shape[1], y, device=logits.device)
        z_y = ops.summation(logits * y_one_hot, axes=(1, ))

        return ops.summation(log_sum_exp - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype))
        self.bias  = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device = device, dtype=dtype).detach()
        self.running_var = init.ones(dim, device = device, dtype=dtype).detach()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            Ex = x.sum(axes = (0, )).reshape((1, self.dim)) / x.shape[0]
            x_sub_Ex = x - Ex.broadcast_to(x.shape)
            Var_x_bais = ops.summation(ops.power_scalar(x_sub_Ex, scalar=2), axes=(0, )) / x.shape[0]
            Var_x_unbais = ops.summation(ops.power_scalar(x_sub_Ex, scalar=2), axes=(0, )) / (x.shape[0] - 1)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * Ex.reshape((self.dim,)).detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * Var_x_bais.detach()
            div_num = ops.power_scalar(Var_x_bais + self.eps, scalar=0.5).reshape((1, self.dim))
            return self.weight.broadcast_to(x.shape) * (x_sub_Ex / div_num.broadcast_to(x.shape)) + self.bias.broadcast_to(x.shape)
        else:
            x_sub_Ex = x - self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            div_num = ops.power_scalar(self.running_var + self.eps, scalar=0.5).reshape((1, self.dim))
            return self.weight.broadcast_to(x.shape) * (x_sub_Ex / div_num.broadcast_to(x.shape)) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype))
        self.bias  = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        E_x = x.sum(axes = (1, )).reshape((x.shape[0], 1)) / self.dim
        x_sub_Ex = x - E_x.broadcast_to(x.shape)
        Var_x = ops.summation(ops.power_scalar(x_sub_Ex, scalar=2), axes=(1, )) / self.dim
        div_num = ops.power_scalar(Var_x + self.eps, scalar=0.5).reshape((x.shape[0], 1))
        return self.weight.broadcast_to(x.shape) * (x_sub_Ex / div_num.broadcast_to(x.shape)) + self.bias.broadcast_to(x.shape)

        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            keep_p = 1 - self.p
            mask = init.randb(*x.shape, p = keep_p)
            return x * mask / keep_p
        else :
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
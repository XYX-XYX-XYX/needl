"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = init.kaiming_uniform(shape=tuple([kernel_size, kernel_size, in_channels, out_channels]), nonlinearity="conv")
        self.weight = Parameter(self.weight, device=device, dtype=dtype)
        fan_in = self.kernel_size * self.kernel_size * self.in_channels
        bound = 1.0 / math.sqrt(fan_in)
        self.bias = init.rand(out_channels, low=-bound, high=bound, device=device, dtype=dtype)
        self.bias = Parameter(self.bias, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_t = x.transpose(tuple([1,3])).transpose(tuple([1,2]))
        out_unbias = ops.conv(x_t, self.weight, self.stride, self.kernel_size // 2) 
        out = out_unbias + self.bias.reshape(tuple([1, 1, 1, self.out_channels])).broadcast_to(out_unbias.shape)
        return out.transpose(tuple([1,3])).transpose(tuple([2,3]))
        ### END YOUR SOLUTION
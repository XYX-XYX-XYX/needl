import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low = -a, high = a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean = 0.0, std = std, **kwargs)
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int = None, fan_out: int = None, shape: tuple = None, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    ### assert nonlinearity == "relu" , "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[0] * shape[1] * shape[3]
        bound = math.sqrt(2) * math.sqrt(3 / fan_in)
        return rand(*shape, low = -bound, high = bound, **kwargs)
    else:
        assert fan_in is not None and fan_out is not None, "fan_in and fan_out must be provided if shape is None"
        bound = math.sqrt(2) * math.sqrt(3 / fan_in)
        return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int = None, fan_out: int = None, shape: tuple = None, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = shape[0] * shape[1] * shape[2]
        fan_out = shape[0] * shape[1] * shape[3]
        std = math.sqrt(2) / math.sqrt(fan_in)
        return randn(*shape, mean = 0, std = std, **kwargs)
    else:
        assert fan_in is not None and fan_out is not None, "fan_in and fan_out must be provided if shape is None"
        std = math.sqrt(2) / math.sqrt(fan_in)
        return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
    ### END YOUR SOLUTION
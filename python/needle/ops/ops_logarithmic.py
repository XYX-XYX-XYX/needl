from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis = 1, keepdims=True)
        exp_z = array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape))
        sum_z = exp_z.sum(axis = 1, keepdims = True)

        logsumexp_z = array_api.log(sum_z) + max_z


        return Z - array_api.broadcast_to(logsumexp_z, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]

        softmax_z = exp(node)

        sum_grad = summation(out_grad, axes=1).reshape((z.shape[0], 1))
        grad = out_grad - softmax_z * sum_grad.broadcast_to(z.shape)
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis = self.axes, keepdims=True)
        exp_z = array_api.exp(Z - array_api.broadcast_to(max_z, Z.shape))
        sum_z = exp_z.sum(axis = self.axes)

        return array_api.log(sum_z) + Z.max(axis=self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]

        if self.axes is not None:
            # 处理指定轴
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            shape = list(z.shape)
            for axis in axes:
                shape[axis] = 1
        else:
        # 处理全部轴 (axes=None)
            shape = [1] * len(z.shape)

        softmax_z = exp(z - node.reshape(shape).broadcast_to(z.shape))

        return out_grad.reshape(shape).broadcast_to(z.shape) * softmax_z
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
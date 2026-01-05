import numpy as np
import pytest
import needle as ndl
from needle import backend_ndarray as nd

# 定义测试设备，ids=["cpu", "cuda"] 使得 pytest -k "cpu" 生效
_DEVICES = [
    nd.cpu(),
    pytest.param(
        nd.cuda(), marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU")
    ),
]

# --- 测试逻辑 ---

# 1. 基础单轴规约
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reduce_sum_basic(device):
    shape = (4, 5, 6)
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    for axis in [0, 1, 2]:
        np_res = _A.sum(axis=axis, keepdims=True)
        nd_res = A.sum(axis=axis, keepdims=True).numpy()
        np.testing.assert_allclose(np_res, nd_res, atol=1e-5, rtol=1e-5)

# 2. 多轴规约 (Tuple Axis)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reduce_sum_multiaxis(device):
    shape = (2, 3, 4, 5)
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    test_cases = [(0, 1), (1, 3), (0, 1, 2, 3), (2, 3)]
    for axis in test_cases:
        # keepdims=True
        np.testing.assert_allclose(_A.sum(axis=axis, keepdims=True), 
                                   A.sum(axis=axis, keepdims=True).numpy(), 
                                   atol=1e-5, rtol=1e-5)
        # keepdims=False
        np.testing.assert_allclose(_A.sum(axis=axis, keepdims=False), 
                                   A.sum(axis=axis, keepdims=False).numpy(), 
                                   atol=1e-5, rtol=1e-5)

# 3. Max 规约
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reduce_max_multiaxis(device):
    shape = (3, 4, 5)
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    test_cases = [(0,), (0, 2), (0, 1, 2)]
    for axis in test_cases:
        np.testing.assert_allclose(_A.max(axis=axis, keepdims=True), 
                                   A.max(axis=axis, keepdims=True).numpy(), 
                                   atol=1e-5, rtol=1e-5)

# 4. 边界情况
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reduce_corner_cases(device):
    shape = (4, 4)
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    # axis=None
    np.testing.assert_allclose(_A.sum(), A.sum(axis=None).numpy(), atol=1e-5, rtol=1e-5)
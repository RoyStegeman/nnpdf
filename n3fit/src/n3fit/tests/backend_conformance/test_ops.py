"""
Numerical conformance of ``backend.ops`` against numpy.

This is the moved-and-generalized version of ``n3fit/tests/test_backend.py``: the checks
are unchanged, but they now run against every backend through the ``backend`` fixture
(the Keras backend and the numpy test double) instead of against a module-level import of
the Keras operations.
"""

import operator

import numpy as np
import pytest

# General parameters
DIM = 7
THRESHOLD = 1e-6

# Arrays to be used during testing
ARR1 = np.random.rand(DIM)
ARR2 = np.random.rand(DIM)
ARR3 = np.random.rand(DIM + 1, DIM)


def are_equal(result, reference, threshold=THRESHOLD):
    """checks the difference between array `reference` and tensor `result` is
    below `threshold` for all elements"""
    res = np.asarray(result)
    assert np.allclose(res, reference, atol=threshold)


def make_checkers(backend):
    """Build the (tensor, array) pairs the checks operate on, for one backend."""
    ops = backend.ops
    return [ops.numpy_to_tensor(ARR1), ops.numpy_to_tensor(ARR2), ops.numpy_to_tensor(ARR3)]


def numpy_check(backend, backend_op, python_op, mode="same"):
    """Receives a backend operation (`backend_op`) and a python operation
    `python_op` and asserts that, applied to two random arrays, the result
    is the same.
    The option `mode` selects the two arrays to be tested and accepts the following
    options:
     - `same` (default): two arrays of the same dimensionality
     - `diff`: first array has one extra dimension that second array
     - `single`: only one array enters the operation
     - (tensor, array): if passed a tuple (backend tensor, numpy array), uses these
        values as tensor and array inputs for the operations
    """
    T1, T2, T3 = make_checkers(backend)
    if mode == "same":
        tensors = [T1, T2]
        arrays = [ARR1, ARR2]
    elif mode == "diff":
        tensors = [T3, T1]
        arrays = [ARR3, ARR1]
    elif mode == "four":
        tensors = [T1, T2, T1, T1]
        arrays = [ARR1, ARR2, ARR1, ARR1]
    elif mode == "twenty":
        tensors = [T1, T2, T1, T1, T1, T1, T1, T1, T1, T1, T1, T2, T1, T1, T1, T1, T1, T1, T1, T1]
        arrays = [
            ARR1, ARR2, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1,
            ARR1, ARR2, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1,
        ]
    elif mode == "ten":
        tensors = [T1, T2, T1, T1, T1, T1, T1, T1, T1, T1]
        arrays = [ARR1, ARR2, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1, ARR1]
    elif mode == "single":
        tensors = [T1]
        arrays = [ARR1]
    elif isinstance(mode, tuple):
        tensors = mode[0]
        arrays = mode[1]

    result = backend_op(tensors)
    reference = python_op(*arrays)
    are_equal(result, reference)


# Test the NNPDF operation vocabulary
def test_c_to_py_fun(backend):
    """The physics operation vocabulary, where the backend provides it.

    The mapping is part of the NNPDF vocabulary rather than a backend primitive, so a
    backend is allowed not to provide it (the numpy test double does, the contract does
    not require it); in that case the check is skipped for this backend.
    """
    if getattr(backend.ops, "c_to_py_fun", None) is None:
        pytest.skip(f"backend {backend.name!r} does not provide c_to_py_fun")
    pytest.importorskip("validphys.convolution")

    # Null function
    numpy_check(backend, backend.ops.c_to_py_fun("NULL"), lambda x: x, "single")
    # Add
    numpy_check(backend, backend.ops.c_to_py_fun("ADD"), operator.add)
    # Ratio
    numpy_check(backend, backend.ops.c_to_py_fun("RATIO"), operator.truediv)
    # ASY
    reference = lambda x, y: (x - y) / (x + y)
    numpy_check(backend, backend.ops.c_to_py_fun("ASY"), reference)
    # SMN
    reference = lambda x, y, z, d: (x + y) / (z + d)
    numpy_check(backend, backend.ops.c_to_py_fun("SMN"), reference, "four")
    # x - abs(y)
    reference = lambda x, y: x - np.abs(y)
    numpy_check(backend, backend.ops.c_to_py_fun("SUBTRACT_ABS"), reference, "same")
    # x + y - abs(z+d)
    reference = lambda x, y, z, d: x + y - np.abs(z + d)
    numpy_check(backend, backend.ops.c_to_py_fun("SUBTRACT_ABSPAIR"), reference, "four")
    # COM
    reference = lambda x, y, z, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t: (
        x + y + z + d + e + f + g + h + i + j
    ) / (k + l + m + n + o + p + q + r + s + t)
    numpy_check(backend, backend.ops.c_to_py_fun("COM"), reference, "twenty")
    # SMT
    reference = lambda x, y, z, d, e, f, g, h, i, j: (
        x + y + z + d + e + f + g + h + i + j
    )
    numpy_check(backend, backend.ops.c_to_py_fun("SMT"), reference, "ten")


# Tests operations
def test_op_multiply(backend):
    numpy_check(backend, backend.ops.op_multiply, operator.mul)


def test_op_log(backend):
    numpy_check(backend, backend.ops.op_log, np.log, mode="single")


def test_flatten(backend):
    T1, T2, T3 = make_checkers(backend)
    ops = backend.ops
    numpy_check(backend, ops.flatten, np.ndarray.flatten, mode=([T3], [ARR3]))


def test_tensor_product(backend):
    T1, T2, T3 = make_checkers(backend)
    ops = backend.ops
    np_result = np.tensordot(ARR3, ARR1, axes=1)
    result = ops.tensor_product(T3, T1, axes=1)
    are_equal(result, np_result)


def test_sum(backend):
    numpy_check(backend, backend.ops.sum, np.sum, mode="single")


def test_nansum(backend):
    """Tests that sums with NaN in the arrays work as expected"""
    ops = backend.ops
    arr_nonan = np.array([1.0, 2.0])
    arr_nan = np.array([2.0, np.nan, 2.0])
    arr_axis_nan = np.array([[3.0, np.nan], [2.0, 6.0]])

    np.testing.assert_allclose(ops.nansum(arr_nonan), np.nansum(arr_nonan))
    np.testing.assert_allclose(ops.nansum(arr_nan), np.nansum(arr_nan))
    np.testing.assert_allclose(ops.nansum(arr_axis_nan, axis=0), np.nansum(arr_axis_nan, axis=0))

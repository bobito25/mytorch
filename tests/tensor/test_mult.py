"""Multiplication tests for the tensor package."""

import numpy as np

from mytorch.tensor import Tensor


def test_tensor_multiplication() -> None:
    a = Tensor([1,2])
    b = Tensor([2,3])
    r = a * b
    r2 = b * a
    expected = np.array([2,6])

    # check dimensions
    assert r.dim == a.dim
    assert r.dim == b.dim
    assert r.dim == r2.dim

    # check result values
    assert (r.data == expected).all()
    assert (r2.data == expected).all()

    # check original values
    assert (a.data == np.array([1,2])).all()
    assert (b.data == np.array([2,3])).all()

    # check memory locations of tensor objects
    assert id(r) != id(a)
    assert id(r) != id(b)
    assert id(r) != id(r2)
    assert id(a) != id(b)

    # check memory locations of data
    assert id(r.data) != id(a.data)
    assert id(r.data) != id(b.data)
    assert id(r.data) != id(r2.data)
    assert id(a.data != b.data)

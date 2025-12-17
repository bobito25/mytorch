"""Multiplication tests for the tensor package."""

import numpy as np

from mytorch.tensor import Tensor, tmult


def test_tensor_multiplication_forward() -> None:
    a = Tensor([[1,0,-1],[2,1,-2]])
    assert a.dim == (2,3)
    b = Tensor([[-1,1],[-2,2],[-3,3]])
    assert b.dim == (3,2)
    r = tmult(a,b,axes=([1],[0]),dim_out=(2,2))
    r2 = tmult(b,a,axes=([0],[1]),dim_out=(2,2))
    expected = np.array([[-4,4],[2,-2]])

    # check dimensions
    assert r.dim == expected.shape
    assert r2.dim == expected.shape

    # check result values
    assert (r.data == expected).all()
    assert (r2.data == expected).all()

    # check original values
    assert (a.data == np.array([[1,0,-1],[2,1,-2]])).all()
    assert (b.data == np.array([[-1,1],[-2,2],[-3,3]])).all()

    # check memory locations of tensor objects
    assert id(r) != id(a)
    assert id(r) != id(b)
    assert id(r) != id(r2)
    assert id(a) != id(b)

    # check memory locations of data
    assert id(r.data) != id(a.data)
    assert id(r.data) != id(b.data)
    assert id(r.data) != id(r2.data)
    assert id(a.data) != id(b.data)

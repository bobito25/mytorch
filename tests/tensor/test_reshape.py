"""Reshape tests for the tensor package."""

import numpy as np

from mytorch.tensor import Tensor


def test_tensor_reshape() -> None:
    a = Tensor([1,2,3,4])
    b = a.reshape((2,2))
    expected = np.array([1,2,3,4]).reshape((2,2))

    # check dimensions
    assert a.dim == (2,2)
    assert b.dim == (2,2)
    assert a == b

    assert b.data.shape == (2,2)
    assert (b.data == expected).all()

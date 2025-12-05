"""Basic tests for the tensor package."""

import numpy as np

from mytorch.tensor import Tensor


def test_tensor_add() -> None:
    a = Tensor([1,2])
    b = Tensor([2,3])
    r = a + b
    assert r.dim == a.dim == b.dim
    assert (r.data == np.array([3,5])).all()

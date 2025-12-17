"""Multiplication tests for the tensor package."""

import numpy as np
import pytest

from mytorch.tensor import Tensor
from mytorch.autograd import GradOperation


def test_tensor_multiplication_forward() -> None:
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


def test_tensor_multiplication_backward() -> None:
    a = Tensor([1,2])
    b = Tensor([2,3], requires_grad=False)
    r: Tensor = a * b
    
    assert r.requires_grad == True

    assert a.grad_op is None
    assert b.grad_op is None

    grad_op = r.grad_op

    assert isinstance(grad_op, GradOperation)
    assert grad_op.op_name == "tensor-elem-mult"
    assert grad_op.backward_fn == Tensor._elem_mul_backward
    assert grad_op.operands == [a, b]

    grad_out = Tensor([1,1])
    r.backward(grad_out)

    assert r.grad == None
    assert a.grad == b
    assert b.grad == None

    a.grad = None  # reset
    grad_out = Tensor([2,3])
    r.backward(grad_out)
    assert a.grad == Tensor([4,9])

    invalid_grad = Tensor([1,1,1])
    with pytest.raises(ValueError, match="Grad must be of same dim as tensor."):
        r.backward(invalid_grad)
    
    a.grad = None
    r.requires_grad = False
    r.backward(grad_out)
    assert a.grad == None

    new_t = Tensor([1,2], requires_grad=True)
    new_t.backward(grad_out)
    assert new_t.grad == grad_out


def test_tensor_deep_multiplication_backward() -> None:
    a = Tensor([1,2])
    b = Tensor([2,3], requires_grad=False)
    c = Tensor([3,4])
    d = Tensor([4,5], requires_grad=False)
    r1: Tensor = a * b
    r2: Tensor = r1 * c
    r3: Tensor = r2 * d
    
    assert r1.requires_grad == True
    assert r2.requires_grad == True
    assert r3.requires_grad == True

    assert a.grad_op is None
    assert b.grad_op is None
    assert c.grad_op is None
    assert d.grad_op is None

    assert isinstance(r1.grad_op, GradOperation)
    assert r1.grad_op.op_name == "tensor-elem-mult"
    assert r1.grad_op.backward_fn == Tensor._elem_mul_backward
    assert r1.grad_op.operands == [a, b]
    assert isinstance(r2.grad_op, GradOperation)
    assert r2.grad_op.op_name == "tensor-elem-mult"
    assert r2.grad_op.backward_fn == Tensor._elem_mul_backward
    assert r2.grad_op.operands == [r1, c]
    assert isinstance(r3.grad_op, GradOperation)
    assert r3.grad_op.op_name == "tensor-elem-mult"
    assert r3.grad_op.backward_fn == Tensor._elem_mul_backward
    assert r3.grad_op.operands == [r2, d]

    grad_out = Tensor([1,1])
    r3.backward(grad_out)

    assert r1.grad == None
    assert r2.grad == None
    assert r3.grad == None
    assert a.grad == Tensor([2,3]) * Tensor([3,4]) * Tensor([4,5])  # a*b*c = Tensor([24,60])
    assert b.grad == None
    assert c.grad == Tensor([2,6]) * Tensor([4,5])  # r1*d == Tensor([8,30])
    assert d.grad == None

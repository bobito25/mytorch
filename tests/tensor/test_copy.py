"""Copy tests for the tensor package."""

from copy import copy, deepcopy

from mytorch.tensor import Tensor
from mytorch.autograd import GradOperation


def test_tensor_copy() -> None:
    a = Tensor([1,2])
    a.grad = Tensor([1,1])
    a.grad_op = GradOperation("tensor-add", Tensor._add_backward, [a, Tensor([3,3])])
    c = copy(a)
    assert a == c
    assert c.grad == None
    assert c.grad_op == None


def test_tensor_deep_copy() -> None:
    a = Tensor([1,2])
    a.grad = Tensor([1,1])
    a.grad_op = GradOperation("tensor-add", Tensor._add_backward, [a, Tensor([3,3])])
    c = deepcopy(a)
    assert a == c
    assert c.grad == Tensor([1,1])
    assert isinstance(c.grad_op, GradOperation)
    assert c.grad_op.op_name == "tensor-add"
    assert c.grad_op.backward_fn == Tensor._add_backward
    assert c.grad_op.operands == [a, Tensor([3,3])]

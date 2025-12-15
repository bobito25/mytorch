"""Serialization tests for the tensor package."""

from mytorch.tensor import Tensor
from mytorch.autograd import GradOperation

def test_tensor_str():
    t1 = Tensor([1,2])
    t2 = Tensor([1,2], requires_grad=True)
    t3 = Tensor([1,2], requires_grad=False)

    assert str(t1) == "Tensor([1 2])"
    assert str(t2) == "Tensor([1 2])"
    assert str(t3) == "Tensor([1 2])"

    t1.grad = Tensor([1,1])
    assert str(t1) == "Tensor([1 2])"

    t1.grad_op = GradOperation("tensor-add", Tensor._add_backward, [Tensor([1,0]), Tensor([0,2])])
    assert str(t1) == "Tensor([1 2])"


def test_tensor_repr():
    t1 = Tensor([1,2])
    t2 = Tensor([1,2], requires_grad=True)
    t3 = Tensor([1,2], requires_grad=False)

    assert repr(t1) == "Tensor([1 2], requires_grad=True)"
    assert repr(t2) == "Tensor([1 2], requires_grad=True)"
    assert repr(t3) == "Tensor([1 2], requires_grad=False)"

    t1.grad = Tensor([1,1])
    assert repr(t1) == "Tensor([1 2], requires_grad=True, grad=Tensor([1 1]))"

    t1.grad_op = GradOperation("tensor-add", Tensor._add_backward, [Tensor([1,0]), Tensor([0,2])])
    assert repr(t1) == "Tensor([1 2], requires_grad=True, grad=Tensor([1 1]), grad_op=GradOperation(tensor-add, [Tensor([1 0], requires_grad=True), Tensor([0 2], requires_grad=True)]))"

"""Mixed deep ops tests for the tensor package."""

from mytorch.tensor import Tensor
from mytorch.autograd import GradOperation


def test_tensor_deep_mixed_backward() -> None:
    a = Tensor([1,2])
    b = Tensor([2,3], requires_grad=False)
    c = Tensor([3,4])
    d = Tensor([4,5], requires_grad=False)
    e = Tensor([2,2])
    r1: Tensor = a * b
    r2: Tensor = c * d
    r3: Tensor = r1 + r2
    r4: Tensor = r3 * e
    
    assert r1.requires_grad == True
    assert r2.requires_grad == True
    assert r3.requires_grad == True
    assert r4.requires_grad == True

    assert a.grad_op is None
    assert b.grad_op is None
    assert c.grad_op is None
    assert d.grad_op is None
    assert e.grad_op is None

    assert isinstance(r1.grad_op, GradOperation)
    assert r1.grad_op.op_name == "tensor-mult"
    assert r1.grad_op.backward_fn == Tensor._mul_backward
    assert r1.grad_op.operands == [a, b]
    assert isinstance(r2.grad_op, GradOperation)
    assert r2.grad_op.op_name == "tensor-mult"
    assert r2.grad_op.backward_fn == Tensor._mul_backward
    assert r2.grad_op.operands == [c, d]
    assert isinstance(r3.grad_op, GradOperation)
    assert r3.grad_op.op_name == "tensor-add"
    assert r3.grad_op.backward_fn == Tensor._add_backward
    assert r3.grad_op.operands == [r1, r2]
    assert isinstance(r4.grad_op, GradOperation)
    assert r4.grad_op.op_name == "tensor-mult"
    assert r4.grad_op.backward_fn == Tensor._mul_backward
    assert r4.grad_op.operands == [r3, e]

    grad_out = Tensor([1,1])
    r4.backward(grad_out)

    assert r1.grad == None
    assert r2.grad == None
    assert r3.grad == None
    assert r4.grad == None
    assert a.grad == Tensor([2,2]) * Tensor([2,3])  # e*b == Tensor([4,6])
    assert b.grad == None
    assert c.grad == Tensor([2,2]) * Tensor([4,5])  # e*d == Tensor([8,10])
    assert d.grad == None
    assert e.grad == Tensor([14,26])  # r3

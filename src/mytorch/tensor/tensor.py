"""
Implements the Tensor class for the MyTorch library.
"""

import numpy as np


class GradOperation:
    def __init__(self, op_name: str, backward_fn: callable, operands: list["Tensor"]):
        self.op_name = op_name
        self.operands = operands
        self.backward_fn = backward_fn

    def backward(self, grad):
        for operand in self.operands:
            if operand.requires_grad:
                new_grad = self.backward_fn(operand, self.operands.copy().remove(operand))
                operand.backward(new_grad * grad)


class Tensor:
    """
    Tensors are multi-dimensional arrays used for numerical computations.
    They track operations for automatic differentiation.
    backward() method accumulates gradients.
    Numpy is used under the hood for efficiency.
    """
    def __init__(self, data: np.ndarray | list, requires_grad: bool = True):
        if isinstance(data, list):
            data = np.array(data)
        self.data = data
        self.dim = data.shape
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_op: GradOperation = None  # operation that created this tensor

    def __add__(self, other: "Tensor"):
        """
        Tensor addition (returns new tensor).

        :param self: First Tensor opperand
        :param other: Second Tensor opperand
        :type other: Tensor
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        if not self.dim == other.dim:
            raise ValueError("Dimensions of operands of tensor addition must be equal:", self.dim, "!=", other.dim)
        t = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        t.grad_op = GradOperation("tensor-add", Tensor._add_backward, [self, other])
        return t

    def __radd__(self, other: "Tensor"):
        return self.__add__(other)
    
    def _add_backward(cls, self, operands):
        return Tensor(np.ones_like(self.data), requires_grad=False)
    
    def __mul__(self, other: "Tensor"):
        """
        Tensor element-wise multiplication (returns new tensor).

        :param self: First Tensor opperand
        :param other: Second Tensor opperand
        :type other: Tensor
        """
        if not isinstance(other, Tensor):
            return NotImplemented
        if not self.dim == other.dim:
            raise ValueError("Dimensions of operands of tensor addition must be equal:", self.dim, "!=", other.dim)
        t = Tensor(np.multiply(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        t.grad_op = GradOperation("tensor-mult", Tensor._mul_backward, [self, other])
        return t
    
    def __rmul__(self, other: "Tensor"):
        return self.__mul__(other)

    def _mul_backward(cls, self, operands):
        if len(operands) != 1:
            raise ValueError("Backward pass for tensor addition with more than 2 operands is not supported.")
        return Tensor(operands[0].data, requires_grad=False)

    def backward(self, grad=None):
        if grad == None:
            if not self.requires_grad:
                raise ValueError("Cannot compute gradient in tensor that has .requires_grad == False")
            if self.grad_op == None:
                raise ValueError("Cannot perform compute gradient when no operation has been performed on this tensor.")
            # keep propagating
            self.grad_op.backward(grad)
        else:
            if not self.requires_grad:
                # stop recursion
                return
            if self.grad_op == None:
                # leaf node
                self.grad = grad

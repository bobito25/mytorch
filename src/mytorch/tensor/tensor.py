"""
Implements the Tensor class for the MyTorch library.
"""

import numpy as np

from mytorch.autograd import GradOperation


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

    def __eq__(self, other: "Tensor"):
        if not isinstance(other, Tensor):
            return NotImplemented
        return (self.data==other.data).all()

    def __str__(self):
        return "Tensor(" + str(self.data) + ")"

    def __repr__(self):
        r = "Tensor("
        r += str(self.data) + ", "
        r += "requires_grad=" + str(self.requires_grad) + ", "
        if self.grad is not None:
            r += "grad=" + str(self.grad) + ", "
        if self.grad_op is not None:
            r += "grad_op=" + str(self.grad_op) + ", "
        r = r.removesuffix(", ")
        r += ")"
        return r

    def __copy__(self):
        return Tensor(self.data, self.requires_grad)

    def __deepcopy__(self, memo):
        t = Tensor(self.data, self.requires_grad)
        t.grad = self.grad
        t.grad_op = self.grad_op
        return t

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

    @classmethod
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
        t.grad_op = GradOperation("tensor-elem-mult", Tensor._elem_mul_backward, [self, other])
        return t

    def __rmul__(self, other: "Tensor"):
        return self.__mul__(other)

    @classmethod
    def _elem_mul_backward(cls, self, operands):
        if len(operands) != 1:
            raise ValueError("Backward pass for tensor addition with more than 2 operands is not supported.")
        return Tensor(operands[0].data, requires_grad=False)

    def backward(self, grad_out=None):
        if grad_out == None:
            if not self.requires_grad:
                raise ValueError("Cannot perform backward in tensor that has .requires_grad == False.")
            if self.grad_op == None:
                raise ValueError("Cannot perform backward when no operation has been performed on this tensor.")
            raise NotImplementedError()
        else:
            if not self.requires_grad:
                # stop recursion
                return
            if grad_out.dim != self.dim:
                raise ValueError("Grad must be of same dim as tensor.")
            if self.grad_op == None:
                # leaf node
                if self.grad == None:
                    self.grad = grad_out
                else:
                    # if grad exists, accumulate
                    self.grad += grad_out
                return
        # keep propagating
        self.grad_op.backward(grad_out)

    def reshape(self, shape: tuple):
        """
        Reshapes data to given shape.

        :param shape: goal shape
        :type shape: tuple
        """
        self.data = self.data.reshape(shape)
        self.dim = shape
        return self

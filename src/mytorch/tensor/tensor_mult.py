"""
Implements Tensor multiplication for arbitrary number of dimensions.
"""

import numpy as np

from .tensor import Tensor


def tmult(x: Tensor, y: Tensor, axes: tuple[list], dim_out: tuple) -> Tensor:
    """
    Tensor multiplication for two tensors x and y.
    The output shape must be given explicitly.
    
    :param x: first operand
    :type x: Tensor
    :param y: second operant
    :type y: Tensor
    :param axes: axes to sum over (tuple that contains two lists which for corresponding axes in x and y)
    :type axes: tuple[list]
    :param dim_out: expected dimensions of the output
    :type dim_out: tuple
    """
    if not isinstance(x, Tensor):
        raise TypeError("first operand of tensor multiplication must be of type Tensor")
    if not isinstance(y, Tensor):
        raise TypeError("second operand of tensor multiplication must be of type Tensor")
    if not isinstance(axes, tuple):
        raise TypeError("axes given for tensor multiplication must be of type tuple")
    if len(axes) != 2:
        raise ValueError("axes tuple given for tensor multiplication must be of length 2")
    x_axes, y_axes = axes
    if not isinstance(x_axes, list):
        raise TypeError("axes given for first operand must be of type list")
    if not isinstance(y_axes, list):
        raise TypeError("axes given for second operand must be of type list")
    if len(x_axes) != len(y_axes):
        raise ValueError("axes given for both operands must be of same length")
    for i in range(len(x_axes)):
        if x_axes[i] < 0 or x_axes[i] >= len(x.dim):
            raise ValueError("axis", x_axes[i], "given at index", i, "for first operand is not valid")
        if y_axes[i] < 0 or y_axes[i] >= len(y.dim):
            raise ValueError("axis", y_axes[i], "given at index", i, "for second operand is not valid")
        if x.dim[x_axes[i]] != y.dim[y_axes[i]]:
            raise ValueError("axes at index", i, "do not match:", x.dim[x_axes[i]], "!=", y.dim[y_axes[i]])

    t = Tensor(np.tensordot(x.data,y.data,axes=axes), requires_grad=x.requires_grad or y.requires_grad)

    if t.dim != dim_out:
        raise ValueError("output of tensor multiplication is not of expected shape", dim_out, "but instead of shape", t.dim)

    return t

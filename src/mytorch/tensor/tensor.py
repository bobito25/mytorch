"""
Implements the Tensor class for the MyTorch library.
"""

import numpy as np


class Tensor:
    """
    Tensors are multi-dimensional arrays used for numerical computations.
    They track operations for automatic differentiation.
    backward() method accumulates gradients.
    Numpy is used under the hood for efficiency.
    """
    def __init__(self, data: np.ndarray | list):
        if isinstance(data, list):
            data = np.array(data)
        self.data = data
        self.dim = data.shape

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
        return Tensor(self.data + other.data)

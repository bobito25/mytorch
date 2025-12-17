"""Top-level package for mytorch."""

from .core import version

from .tensor import Tensor, tmult

__all__ = [version, Tensor, tmult]

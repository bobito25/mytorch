"""Core utilities for mytorch."""

__version__ = "0.1.0"


def version() -> str:
    """Return the current package version."""
    return __version__


def example(x: float, y: float) -> float:
    """Return a trivial computation used in simple tests."""
    return x + y

"""Basic tests for the mytorch package."""

from mytorch import example, version


def test_example_adds_values() -> None:
    assert example(1.5, 2.5) == 4.0


def test_version_string() -> None:
    assert version().startswith("0.1")

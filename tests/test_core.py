"""Basic tests for the mytorch package."""

from mytorch import version


def test_version_string() -> None:
    assert version().startswith("0.1")

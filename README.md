# mytorch

Minimal scaffold for the `mytorch` Python package.

## Structure

- `pyproject.toml` – build metadata using PEP 621 and setuptools.
- `src/mytorch/` – package sources.
- `tests/` – pytest-compatible smoke tests.

## Publishing

1. Update `pyproject.toml` metadata (version, authors, classifiers) as needed.
2. Build sdist/wheel with `python -m build`.
3. Upload to PyPI or other index via `twine upload dist/*`.

## Development

Install dev dependencies then run `python -m pytest` to exercise the basic smoke tests.

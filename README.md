# jaxtyping2doc
![Tests](https://github.com/jonasjuerss/jaxtyping2doc/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/jonasjuerss/jaxtyping2doc/graph/badge.svg?token=lA4UfX1ScI)](https://codecov.io/gh/jonasjuerss/jaxtyping2doc)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

While I like the idea to annotate the dimensions of torch/jax/numpy/... arrays using [jaxtyping](https://github.com/patrick-kidger/jaxtyping), I want the dimensions to show up in my docstrings, particularly in the pop-ups generated in PyCharm/VSCode. This tool automatically adds and updates tensor dimensions in docstrings based on jaxtyping hints.

This project will not be actively maintained by me, but I am happy to review and accept pull requests.

## Usage as pre-commit hook
```yaml
repos:
-   repo: https://github.com/jonasjuerss/jaxtyping2doc
    rev: v0.0.3  # Use the latest version here
    hooks:
    -   id: jaxtype2doc
```
When using jaxtype2doc together with a formatting pre-commit hook like [black](https://github.com/psf/black), make sure to list the reformatting **after** jaxtype2doc.

## Usage as command
### Installation
```bash
pip install git+https://github.com/jonasjuerss/jaxtyping2doc.git
```
### Usage
```bash
jaxtype2doc file1.py file2.py ...
```

## Contributing
If you experience any issues while using jaxtype2doc, I am happy to review pull-requests.

```bash
git clone <your fork>
pip install -e .[testing]
pre-commit install
```

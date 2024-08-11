# jaxtyping2doc
![Tests](https://github.com/jonasjuerss/jaxtyping2doc/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/jonasjuerss/jaxtyping2doc/graph/badge.svg?token=lA4UfX1ScI)](https://codecov.io/gh/jonasjuerss/jaxtyping2doc)

While I like the idea to annotate the dimensions of torch/jax/numpy/... arrays using [jaxtyping](https://github.com/patrick-kidger/jaxtyping), I want the dimensions to show up in my docstrings, particularly in the pop-ups generated in PyCharm/VSCode. This tool automatically adds and updates tensor dimensions in docstrings.



## Usage

````bash
jaxtype2doc file1.py file2.py ...
````

## Installation
### Create virtual environment
```bash
python -m venv venv
```
### Activate virtual environment
#### Linux
```bash
source venv/bin/activate
```
#### Windows
```bash
venv\Scripts\activate
```
### Installation

#### For usage
```bash
pip install .
```

#### For development
```bash
pip install -e .[testing]
pre-commit install
```

import typing

import numpy as np
from torch import Tensor
from jaxtyping import Float, Bool
import jaxtyping
from typing import Dict


test_array: Bool[np.ndarray, "batch_size 3 5"]
"""An annotated np attribute"""

NAME: str = "Test"
"""This is NOT a tensor"""


def basic_calc(
    a: Float[Tensor, " *batch_dims 10"]
) -> jaxtyping.Int[Tensor, " *batch_dims"]:
    # Hellow
    return a.sum(-1).round().int()


def basic_calc_doc(
    a: Float[Tensor, " *batch_dims 10"], b: str
) -> jaxtyping.Int[Tensor, " *batch_dims"]:
    """
    Lol
    :param a: a tensor
    :param b: a string
    :return: [20, 3] tensor sth
    """
    return a.sum(-1).round().int()


def basic_calc_doc_tuple(
    a: Float[Tensor, " *batch_dims 10"], b: str
) -> tuple[str, jaxtyping.Int[Tensor, " *batch_dims"]]:
    """
    This is a test ``and`` another.

    :param a: variable a
    :param b: variable b
    :return: sth
    """
    return b, a.sum(-1).round().int()


def basic_calc_doc_list(
    a: Float[Tensor, " *batch_dims 10"], b: str
) -> typing.List[jaxtyping.Int[Tensor, " *batch_dims"]]:
    """
    This is a test ``and`` another.

    :param a: [*batch_dims, 10] float tensor variable a
    :param b: variable b
    :return: sth
    """
    return [a.sum(-1).round().int()]


def basic_calc_doc_dict(
    a: Float[Tensor, " *batch_dims 10"], b: str
) -> Dict[str, jaxtyping.Int[Tensor, " *batch_dims"]]:
    """
    :param a: variable a
    :param b: variable b
    :return: sth
    :raise: NotImplementedError: test
    """
    return a.sum(-1).round().int()


def one_line_comment(
    x: Float[Tensor, " *batch_dims 10"]
) -> Float[Tensor, " *batch_dims 10"]:
    """This is just a quick one-line description without documenting the parameters."""
    pass

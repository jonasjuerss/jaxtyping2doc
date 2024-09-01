"""
Some general info about this class
"""
import abc
from enum import Enum

import jaxtyping
import numpy as np
from jaxtyping import Int, Bool
from torch import Tensor


class MyEnum(Enum):
    """This is just some enum"""
    ONE = 1
    TWO = 2
    """This is number two"""
    THREE = 3

class TestClass(abc.ABC):
    """
    This is a docstring for the entire test class
    """

    # TODO this should also be annotated automatically in the future
    attr2: Bool[np.ndarray, " *any_dims"]
    """"""
    def __init__(self, attr1: Int[Tensor, "dims1 dims2"], b: str) -> None:
        """
        The constructor

        :param attr1: attribute 1
        """
        # TODO this should also be annotated automatically in the future
        self.attr1: Bool[np.ndarray, "batch_size 3 5"]
        """An annotated np attribute"""

    def no_docstring(self, a: str, b: jaxtyping.Float[np.ndarray, " *dims"]) -> None:
        print(a)

    def meth(self, b: jaxtyping.Float[np.ndarray, " *dims"]) ->\
            jaxtyping.Float[np.ndarray, " *dims"]:
        """
        :param b: [*dims] float array a tensor
        :returns: [*dims] float array the same tensor
        """
        return b

    @staticmethod
    def static_meth(b: jaxtyping.Float[np.ndarray, " *dims"]) ->\
            jaxtyping.Float[np.ndarray, " *dims"]:
        """
        :param b: [*dims] float array a tensor
        :returns: [*dims] float array the same tensor
        """
        return b


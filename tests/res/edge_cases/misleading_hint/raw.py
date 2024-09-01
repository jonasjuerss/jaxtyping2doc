import typing
import torch
from jaxtyping import Int
from torch import Tensor

Float = typing.Annotated


def fun(a: Int[Tensor, "batch_size"], b: Float[Tensor, "batch_size"]):
    """
    Test message

    :param a: jaxtyping annotated
    :param b: not jaxtyping annotated
    :returns:
    """
    pass


if __name__ == "__main__":
    fun(torch.arange(1), torch.arange(1))

import jaxtyping
import numpy as np
from jaxtyping import Float
from torch import Tensor


def basic_calc_doc(
    a: Float[Tensor, " *batch_dims 10"], b: Float[np.ndarray, " *batch_dims 10"]
) -> jaxtyping.Int[Tensor, " *batch_dims"]:
    """
    Lol
    :param a: [*batch_dims, 10] float tensor incomplete/wrong documentation
    :param b: [*batch_dims, 10] float array incomplete documentation
    :returns: [*batch_dims] int tensor wrong documentation
    """
    return a.sum(-1).round().int()

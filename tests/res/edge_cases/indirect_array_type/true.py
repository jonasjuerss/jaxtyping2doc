import numpy as np
import torch
from jaxtyping import Int

def fun(a: Int[np.ndarray, "batch_size"], b: Int[torch.Tensor, "batch_size"]):
    """
    Test message

    :param a: [batch_size] int array numpy
    :param b: [batch_size] int tensor torch
    :returns:
    """
    pass

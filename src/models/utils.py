import jax.numpy as np
from jax import random
from flax import linen as nn


def stochastic_depth(key, input, p, mode, training=True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("drop probability has to be between 0 and 1, but got {}".format(p))
    if mode not in ["batch", "row"]:
        raise ValueError("mode has to be either 'batch' or 'row', but got {}".format(mode))
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim

    noise = random.bernoulli(key, p=p, shape=size)
    noise = noise/survival_rate
    return input * noise


class StochasticDepth(nn.Module):
    """
    See :func:`stochastic_depth`.
    """
    p: float
    mode: str

    @nn.compact
    def __call__(self, input, training):
        if training:
            key = self.make_rng('dropout')
        else:
            key = None
        return stochastic_depth(key, input, self.p, self.mode, training)

    # def __repr__(self) -> str:
    #     tmpstr = self.__class__.__name__ + '('
    #     tmpstr += 'p=' + str(self.p)
    #     tmpstr += ', mode=' + str(self.mode)
    #     tmpstr += ')'
    #     return tmpstr


class Identity(nn.Module):
    @nn.compact
    def __call__(self, input):
        return input

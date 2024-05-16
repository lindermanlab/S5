"""Quantization utilities for qSSMs, using the `aqt` library.

Note:
For both quant_dot_for_hadamard and quant_dot_for_dot, the returned functions
do not support batch dimensions. To add batch dimensions, simply call jax.vmap
on the returned functions.

Example usage:
```python
int8_config = fully_quantized(fwd_bits=8, bwd_bits=8)
q_had = quant_dot_for_hadamard(int8_config)
q_dot = quant_dot_for_dot(int8_config)
q_had(jnp.ones(10,), jnp.ones(10,))
```
"""
from aqt.jax.v2.aqt_dot_general import CalibrationMode
from functools import partial
from typing import Optional, Union
import aqt.jax.v2.config as aqt_config
import jax.numpy as np
import jax


fully_quantized = partial(
    aqt_config.fully_quantized,
    calibration_mode=CalibrationMode.ALL_AXES, use_stochastic_rounding=False,
)


def q_dot_maybe(lhs_bits: Optional[int], rhs_bits: Optional[int]):
    if lhs_bits is None and rhs_bits is None:
        return np.dot
    else:
        precision = (lhs_bits, rhs_bits)
        bwd_bits = max([e for e in precision if e is not None])
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=bwd_bits)
        return quant_dot_for_dot(dot_general)


def q_had_maybe(lhs_bits: Optional[int], rhs_bits: Optional[int]):
    if lhs_bits is None and rhs_bits is None:
        return np.multiply
    else:
        precision = (lhs_bits, rhs_bits)
        bwd_bits = max([e for e in precision if e is not None])
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=bwd_bits)
        return quant_dot_for_hadamard(dot_general)


def quant_dot_for_hadamard(dot_general):
    """Generate a jitted general_dot function to be used for hadamard products.
    Note that this function does not support batch dimensions. All dimensions will
    be used for calibration in the quantization."""
    def _dot(a, b):
        contr_dims = ((), ())  # hadamard has no contracting dims
        batch_dims = (tuple(range(a.ndim)), tuple(range(b.ndim)))  # use all dims as batch dims
        return dot_general(a, b, (contr_dims, batch_dims))
    return jax.jit(_dot)


def quant_dot_for_dot(general_dot):
    """Generate a jitted general_dot function to be used for dot products.
    Will contract on the last dimension of a, and the first dimension of b.
    This means that there are no batch dimensions, and all dimensions will be used
    for calibration in the quantization."""
    def _dot(a, b):
        # contr_dims = ((a.ndim-1,), (1,))  # batched version (not used)
        # batch_dims = ((0,), (0,))  # batched version (not used)
        contr_dims = ((a.ndim-1,), (0,))
        batch_dims = ((), ())
        return general_dot(a, b, (contr_dims, batch_dims))
    return jax.jit(_dot)

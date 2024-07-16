import pytest

import numpy.testing  # provides more informative failure messages
import jax
import jax.numpy as jnp
import jax.random as jr

from s5dev.models.simple_lm import create_block

DEFAULT_RNG = jr.PRNGKey(21350)

DEFAULT_S5OPERATOR_KWARGS = dict(
    ssm_size = 4,
    ssm_blocks = 1,
    order = 2,
    inner_factor = 1,
    drop_rate = 0.0,
    filter_cls = 'hyena_S5',
    short_filter_order = 0,  # must be 0 in order to compare __call__ and step implementations
    filter_args = dict(
        C_init = "complex_normal",
        dt_min = 0.001,
        dt_max = 0.1,
        conj_sym = False,
        clip_eigs = True,
        activation = "gelu",
    )
)


def test_block_step(layer='s5_operator', batch_size=64, seq_len=8, d_input=12, atol=1e-4, rtol=1e-3):
    """Test Block autoregressive generation via `step` implementation.
    
    Only applicable for layers with `step` function implemented, e.g. layer='s5_operator'.
    """

    init_rng, data_rng = jr.split(DEFAULT_RNG)

    input_seq = jr.normal(data_rng, shape=(batch_size, seq_len, d_input))
    init_residual = None

    # Initialize Block
    if layer == 's5_operator':
        layer_kwargs = DEFAULT_S5OPERATOR_KWARGS
    
    model = create_block(
        d_input, n_layer=1, l_max=seq_len, 
        layer=layer, layer_kwargs=layer_kwargs,
        resid_dropout1=0.0, resid_dropout2=0.0,
    )
    model_variables = model.init(init_rng, jnp.zeros_like(input_seq), training=False)

    # Generate reference output, using __call__
    ys_refr, residuals_refr = model.apply(model_variables, input_seq, residual=init_residual, training=False)

    # -----------------------------------------------------------------------------------
    # Generate output autoregressively
    init_state = jnp.zeros((batch_size, layer_kwargs['ssm_size']), dtype="complex")  # dtype="complex": allow jax backend to set correct precision

    def _step(mixer_state, inpt):
        new_mixer_state, output, residual = \
            model.apply(model_variables, mixer_state, inpt, method='step', residual=init_residual)
        return new_mixer_state, (output, residual)
    
    x, (ys_test_T, residuals_test_T) = jax.lax.scan(
        _step, init_state, jnp.transpose(input_seq, (1,0,2))
    )  # *_test_T shape: (seq_len, bsz, d_input)

    ys_test = jnp.transpose(ys_test_T, (1,0,2))  # now, (bsz, seq_len, d_input)
    residuals_test = jnp.transpose(residuals_test_T, (1,0,2))  # now, (bsz, seq_len, d_input)

    # -----------------------------------------------------------------------------------

    y_err = jnp.abs(ys_refr-ys_test)
    res_err = jnp.abs(residuals_refr-residuals_test)
    msg = ""
    for lbl, err in (('y', y_err), ('res', res_err)):
        msg += (f"{lbl}: (mean: {jnp.mean(err):.1e}, "
                + f"P50: {jnp.percentile(err, 50):.1e}, "
                + f"P90: {jnp.percentile(err, 90):.1e}). ")
    print(msg)

    numpy.testing.assert_allclose(ys_refr, ys_test, atol=atol, rtol=rtol)
    numpy.testing.assert_allclose(residuals_refr, residuals_test, atol=atol, rtol=rtol)

    assert x.dtype == jnp.array([], dtype="complex").dtype

"""
We test everything in double-precision mode (set at top of file)
to control for errors and lack of equivalency due to numerical imprecision.

Identified sources of numerical imprecision are noted in the respective test functions.

For more verbose messages, call with '-s' flag, i.e.
    pytest -s <this_file.py>
"""

import pytest

import numpy.testing  # provides more informative failure messages
import jax
import jax.numpy as jnp
import jax.random as jr

from s5dev.models.simple_lm import create_block, LMBackbone

jax.config.update('jax_enable_x64', True)

DEFAULT_RNG = jr.PRNGKey(21350)

IDENTITY_KWARGS = dict(
    order = 2,
    inner_factor = 1,
    drop_rate = 0.0,
    short_filter_order = 0,  # must be 0 in order to compare __call__ and step implementations
)

DEFAULT_S5OPERATOR_KWARGS = dict(
    ssm_size = 4,
    ssm_blocks = 1,
    order = 2,
    inner_factor = 1,
    drop_rate = 0.0,
    short_filter_order = 0,  # must be 0 in order to compare __call__ and step implementations
    filter_cls = 'hyena_S5',
    filter_args = dict(
        C_init = "complex_normal",
        dt_min = 0.001,
        dt_max = 0.1,
        conj_sym = False,
        clip_eigs = True,
        activation = "gelu",
    )
)


@pytest.mark.parametrize(
    "layer, layer_kwargs", [("identity", IDENTITY_KWARGS), ("s5_operator", DEFAULT_S5OPERATOR_KWARGS)]
)
def test_block_step(layer, layer_kwargs, batch_size=64, seq_len=8, d_input=3):
    """Test Block autoregressive generation via `step` implementation.
    
    Only applicable for layers with `step` function implemented, e.g. layer='s5_operator'.
    """

    init_rng, data_rng = jr.split(DEFAULT_RNG)

    input_seq = jr.normal(data_rng, shape=(batch_size, seq_len, d_input))
    init_residual = None

    # Initialize Block
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
    if layer == 's5_operator':
        init_state = jnp.zeros((batch_size, layer_kwargs['ssm_size']), dtype=complex)
    elif layer == 'identity':
        init_state = None

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

    numpy.testing.assert_allclose(ys_refr, ys_test)
    numpy.testing.assert_allclose(residuals_refr, residuals_test)

    if layer == 's5_operator':
        assert jnp.issubdtype(x.dtype, complex)


@pytest.mark.parametrize(
    "layer, layer_kwargs, n_layer", [
        ("identity", IDENTITY_KWARGS, 1),
        ("identity", IDENTITY_KWARGS, 10),
        ("s5_operator", DEFAULT_S5OPERATOR_KWARGS, 1),
        ("s5_operator", DEFAULT_S5OPERATOR_KWARGS, 2),
        ("s5_operator", DEFAULT_S5OPERATOR_KWARGS, 5),
        ("s5_operator", DEFAULT_S5OPERATOR_KWARGS, 10),
    ]
)
def test_lmbackbone_step(layer, layer_kwargs, n_layer, batch_size=64, seq_len=8, d_input=3):
    """Test Block autoregressive generation via `step` implementation.
    
    Only applicable for layers with `step` function implemented, e.g. layer='s5_operator'.

    Identified sources of numerical imprecision
    -------------------------------------------
    LMBackbone models with multiple S5Operator layers were found to have discrepancies
    between the __call__ and step functions. These are posited to be due to imprecisions
    in propogating the complex S5SSM state (even in double-precision), because
        - When stepping through the `__call__` vs. `step` implementations of each Block layer,
          discrepancy is introduced after the second Block's call to the sequence mixer,
                new_ssm_state, hidden_state = self.mixer.step(ssm_state, hidden_state)
        - However, no such error is introduced when using `self.mixer = IdentitySSM`,
          which is a pass-through module that does not use state.
    This can result in small large spurious errors, which dictate the tolerance values.
    However, these errors are typically 1-2% of the elements, and so a the 90th percentile
    error values are provided next to each parameterization for a more robust sense of the
    true error tolerance (see inline comments above, denoted 'P90')

    """

    init_rng, data_rng = jr.split(DEFAULT_RNG)

    input_seq = jr.normal(data_rng, shape=(batch_size, seq_len, d_input))

    # Initialize Block
    model = LMBackbone(
        d_input, n_layer=n_layer, d_inner=4*d_input,
        layer=layer, layer_kwargs=layer_kwargs, l_max=seq_len,
        resid_dropout=0.0, embed_dropout=0.0,
    )
    model_variables = model.init(init_rng, jnp.zeros_like(input_seq), training=False)

    # Generate reference output, using __call__
    ys_refr, _ = model.apply(model_variables, input_seq, training=False)

    # -----------------------------------------------------------------------------------
    # Generate output autoregressively
    if layer == 's5_operator':
        init_state = jnp.zeros((n_layer, batch_size, layer_kwargs['ssm_size']), dtype=complex)
    elif layer == 'identity':
        init_state = jnp.zeros((n_layer, batch_size))
    
    x, ys_test_T = jax.lax.scan(
        lambda state, inpt: model.apply(model_variables, state, inpt, method='step'),
        init_state, jnp.transpose(input_seq, (1,0,2))
    )  # ys_test_T shape: (seq_len, bsz, d_input)

    ys_test = jnp.transpose(ys_test_T, (1,0,2))  # now, (bsz, seq_len, d_input)

    # -----------------------------------------------------------------------------------

    y_err = jnp.abs(ys_refr-ys_test)
    msg = (f"(mean: {jnp.mean(y_err):.1e}, "
          + f"P50: {jnp.percentile(y_err, 50):.1e}, "
          + f"P90: {jnp.percentile(y_err, 90):.1e}). ")
    print(msg)

    numpy.testing.assert_allclose(ys_refr, ys_test)

    if layer == 's5_operator':
        assert jnp.issubdtype(x.dtype, complex)

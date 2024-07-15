"""
To test in double precision, set
    export JAX_ENABLE_X64=True
    pytest <this_file>.py

To view more verbose messages, call
    pytest -s <this_file>.py
"""

import pytest

import einops
import numpy.testing  # provides more informative failure messages
import jax
import jax.numpy as jnp
import jax.random as jr

from s5dev.models.s5 import (
    apply_ssm,
    discretize_zoh,
    init_S5SSM,
    S5Operator,
    S5SSM
)

DEFAULT_RNG = jr.PRNGKey(55553)

# corresponds to layer_kwargs.filter_args settings in config files
DEFAULT_S5SSM_KWARGS = dict(
    C_init = "complex_normal",
    dt_min = 0.001,
    dt_max = 0.1,
    conj_sym = False,
    clip_eigs = True,
    activation = "gelu",
)


C_INIT_METHODS = ["trunc_standard_normal", "lecun_normal", "complex_normal"]


def reshape_for_s5ssm(arr, n_heads=1, n_blocks=1):
    """Reshape batched sequences for S5SSM 
    
    input:  shape (batch_size, seq_len, d_input)
    output: shape (batch_size, n_heads, d_input, n_blocks, seq_len)

    """
    arr = einops.rearrange(arr, "b d l -> b l d")
    return jnp.tile(arr[:,None,:,None,:], reps=(1, n_heads, 1, n_blocks, 1))


def simple_apply_ssm(A, B, C, D, init_state, input_seq, conj_sym):
    """Simple for-loop implemention of apply input sequence to SSM.
    
    Args:
        A: Array[complex64], shape (_d_state,)
        B: Array[float32], shape (_d_state, d_input)
        C: Array[float32], shape (d_input, _d_state)
        D: Arrat[float32], shape (d_input,)
        init_state: Array[complex64], shape (d_state,)
        input_seq: Array[float32], shape (seq_len, d_input)
        conj_sym: bool. If True, scale state emissions Cx by 2.

    Return:
        x: Array[complex64], shape (d_state)
        output_seq: Array[float32], shape (seq_len, d_input)
    """

    fctr = 2 if conj_sym else 1

    output_seq = []
    x = init_state
    for u in input_seq:
        x = A * x + B @ u
        y = fctr * (C @ x).real + D * u
        output_seq.append(y)

    return x, jnp.stack(output_seq, axis=0)


@pytest.mark.parametrize("C_init", C_INIT_METHODS)
@pytest.mark.parametrize("conj_sym", [True, False])
@pytest.mark.parametrize("clip_eigs", [True, False])
def test_apply_ssm(
    C_init, conj_sym, clip_eigs, batch_size=8, seq_len=128, d_input=16, d_state=64, n_blocks=4,
    atol=1e-4, rtol=1e-1
):
    """Evaluate equivalence of `apply_ssm` using parallel scan to simply for loop implementation."""

    rng, init_rng, data_rng = jr.split(DEFAULT_RNG, num=3)

    ssm_kwargs = DEFAULT_S5SSM_KWARGS | {
        "C_init": C_init, "conj_sym": conj_sym, "clip_eigs": clip_eigs
    }

    # Generate random input sequence
    input_seq = jr.normal(rng, shape=(batch_size, seq_len, d_input))

    # Initialize SSM parameters
    # TODO Refactor and functionalize the init functions of SSM parameters.
    model = init_S5SSM(d_input, d_state, n_blocks, ssm_kwargs)
    expanded_input_seq = reshape_for_s5ssm(input_seq)  # shape (batch_size, n_heads, d_input, n_blocks, seq_len)
    model_variables = model.init(init_rng, jnp.zeros_like(expanded_input_seq), training=False)

    Lambda = model_variables['params']['Lambda_re'] + 1j * model_variables['params']['Lambda_im']
    if ssm_kwargs['clip_eigs']:
        Lambda = jnp.clip(Lambda.real, None, -1e-4) + 1j * Lambda.imag

    B_tilde = model_variables['params']['B'][...,0] + 1j*model_variables['params']['B'][...,1]
    C_tilde = model_variables['params']['C'][...,0] + 1j*model_variables['params']['C'][...,1]
    D = model_variables['params']['D']

    discrete_step = jnp.exp(model_variables['params']['log_step'][:,0])

    Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, discrete_step)

    # -----------------------------------------------------------------------------------
    init_state = jnp.zeros((batch_size, Lambda.shape[-1]), dtype="complex")  # dtype="complex": allow jax backend to set correct precision
    xs_refr, ys_refr = jax.vmap(
        simple_apply_ssm, in_axes=(None,None,None,None,0,0,None),
    )(Lambda_bar, B_bar, C_tilde, D, init_state, input_seq, conj_sym)

    xs_test, ys_test = jax.vmap(
        apply_ssm, in_axes=(0,None,None,None,None,None),
    )(input_seq, Lambda_bar, B_bar, C_tilde, D, conj_sym)
   
    # -----------------------------------------------------------------------------------
    x_err = jnp.abs(xs_refr-xs_test)
    y_err = jnp.abs(ys_refr-ys_test)
    msg = f"({C_init[:14]}, {conj_sym}, {clip_eigs}) error\t"
    for lbl, err in (('xs', x_err), ('ys', y_err)):
        msg += (f"{lbl}: (mean: {jnp.mean(err):.1e}, "
                + f"P50: {jnp.percentile(err, 50):.1e}, "
                + f"P90: {jnp.percentile(err, 90):.1e}). ")
    print(msg)

    numpy.testing.assert_allclose(xs_refr, xs_test, atol=atol, rtol=rtol)
    numpy.testing.assert_allclose(ys_refr, ys_test, atol=atol, rtol=rtol)

    assert xs_refr.dtype == jnp.array([], dtype="complex").dtype


@pytest.mark.parametrize("C_init", C_INIT_METHODS)
@pytest.mark.parametrize("conj_sym", [True, False])
@pytest.mark.parametrize("clip_eigs", [True, False])
def test_s5ssm_step(
    C_init, conj_sym, clip_eigs, batch_size=64, seq_len=8, d_input=12, d_state=4, n_blocks=1, atol=1e-8, rtol=1e-7
):
    """Test S5SSM autoregressive generation via `step` implementation.
    
    Given an input sequence, the autoregressively generated output sequence should be
    the same as the output sequence produced in parallel by applying `__call__`.

    Hyperparmeter settings impact the amount of propgated error. For example, we expect
    longer sequences and larger state dimensions to increase amount of error between the
    parallel `__call__` and the autoregressive `step`.
    """

    init_rng, data_rng = jr.split(DEFAULT_RNG, num=3)

    ssm_kwargs = DEFAULT_S5SSM_KWARGS | {
        "C_init": C_init, "conj_sym": conj_sym, "clip_eigs": clip_eigs
    }

    # Generate random input sequence
    input_seq = jr.normal(data_rng, shape=(batch_size, seq_len, d_input))
    input_seq = reshape_for_s5ssm(input_seq)  # shape (bsz, n_heads, d_input, n_blocks, seq_len)

    # Initialize S5SSM layer
    model = init_S5SSM(d_input, d_state, n_blocks, ssm_kwargs)
    model_variables = model.init(init_rng, jnp.zeros_like(input_seq), training=False)

    # Generate reference output, using parallel scan
    ys_refr = model.apply(model_variables, input_seq)

    # Generate output autoregressively
    init_state = jnp.zeros((batch_size, model.P), dtype="complex")  # dtype="complex": allow jax backend to set correct precision
    x, ys_test_T = jax.lax.scan(
        lambda state, u_T: model.apply(model_variables, state, u_T.T, method='step'),
        init_state, input_seq.T
    )  # ys_test_T shape: (seq_len, bsz, n_heads, H, n_blocks)

    ys_test = jnp.transpose(ys_test_T, (1,2,3,4,0))  # now, (bsz, 1, d_input, 1, seq_len)

    # To see these statements in console, run `pytest -s ...`
    err = jnp.abs(ys_refr-ys_test)
    print(f"({C_init[:14]}, {conj_sym}, {clip_eigs}) error\t"
        + f"mean: {jnp.mean(err):.1e}, "
        + f"P50: {jnp.percentile(err, 50):.1e}, "
        + f"P90: {jnp.percentile(err, 90):.1e}."
    )
    numpy.testing.assert_allclose(ys_refr, ys_test, atol=atol, rtol=rtol)

    assert x.dtype == jnp.array([], dtype="complex").dtype
    

@pytest.mark.parametrize("inner_factor, atol, rtol", [(1, 1e-8, 1e-7), (3, 1e-4, 1e-2)])
def test_s5operator_step(
    inner_factor, atol, rtol,
    batch_size=64, seq_len=8, d_input=12,
    ssm_size=4, ssm_blocks=1,
):
    """Test S5Operator autoregressive generation via `step` implementation.
    
    Given an input sequence, the autoregressively generated output sequence should be
    the same as the output sequence produced in parallel by applying `__call__`.

    Note: Equivalence between __call__ and step is only applicable for hyena_order = 2,
    i.e. no hyena recurrence. If hyena_order > 2, then there is a for-loop that is operating
    on a (mixed) projection that has been convolved with itself. This is not possible in
    the AR mode, so it is not applicable to compare to hyena_order > 2.

    Hyperparmeter settings impact the amount of propgated error. For example, we expect
    longer sequences and larger state dimensions to increase amount of error between the
    parallel `__call__` and the autoregressive `step`.
    """

    init_rng, data_rng = jr.split(DEFAULT_RNG)

    # Generate random input sequence
    input_seq = jr.normal(data_rng, shape=(batch_size, seq_len, d_input))

    # Initialize S5Operator module
    model = S5Operator(
        d_input, n_layer=1, l_max=seq_len,
        filter_cls='hyena_S5', ssm_size=ssm_size, ssm_blocks=ssm_blocks, filter_args=DEFAULT_S5SSM_KWARGS,
        order=2, inner_factor=inner_factor, drop_rate=0.0,
        short_filter_order=0,  # short_filter_order=0 required for comparison
        )
    model_variables = model.init(init_rng, jnp.zeros_like(input_seq), training=False)

    # Generate reference output, using parallel scan
    ys_refr = model.apply(model_variables, input_seq, training=False)

    # -----------------------------------------------------------------------------------
    # Generate output autoregressively
    init_state = jnp.zeros((batch_size, ssm_size), dtype="complex")  # dtype="complex": allow jax backend to set correct precision
    x, ys_test_T = jax.lax.scan(
        lambda state, u: model.apply(model_variables, state, u, method='step'),
        init_state, jnp.transpose(input_seq, (1,0,2))
    )  # ys_test_T shape: (seq_len, bsz, d_input)

    ys_test = jnp.transpose(ys_test_T, (1,0,2))  # now, (bsz, seq_len, d_input)

    # -----------------------------------------------------------------------------------

    numpy.testing.assert_allclose(ys_refr, ys_test, atol=atol, rtol=rtol)

    assert x.dtype == jnp.array([], dtype="complex").dtype
    

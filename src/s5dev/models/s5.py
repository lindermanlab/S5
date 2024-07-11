from functools import partial
import math

from einops import rearrange, repeat
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
import jax
import jax.numpy as np
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag

from s5dev.models.hyena import Activation, mul_sum
from s5dev.models.ssm_init import (
    init_CV,
    init_log_steps,
    init_VinvB,
    make_DPLR_HiPPO,
    trunc_standard_normal
)


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence.

    Should be called like
        _, xs = jax.lax.associative_scan(binary_operator, (A_elements, B_elements))
    Assumes diagonal A matrix.

    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)

    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(input_seq, Lambda_bar, B_bar, C_tilde, D, conj_sym):
    """Compute the output sequence of a discretized SSM using a parallel scan.

    Assumes initial state is vector of all 0's.

    Args:
        input_seq : Array[float32], shape (L,H). Input sequence
        Lambda_bar: Array[complex64], shape (P,). Discretized diagonal state matrix.
        B_bar     : Array[complex64], shape (P,H). Discretized input matrix
        C_tilde   : Array[complex64], shape (H,P). State emissions matrix
        D         : Array[float32], shape (H,). Diagonal feedthrough matrix.
        conj_sym  : bool. If True, indicates that conjugate symmetry is enforced by halving
            state representation. This implies that when taking the real portion of the
            state emission, it needs to be multipled by a factor of 2.
        bidirectional: bool. If True, use bidirectional setup; C_tilde is shape (H, 2P).
            TODO Needs to be implemented
    Returns:
        x: Array[complex64], shape (P,). Last state
        ys Array[float32], shape (L,H). Emissions with feedthrough
    """
    
    # Apply state dynamics using parallel scan
    Lambda_elements = np.tile(Lambda_bar, reps=(len(input_seq), 1))  # shape (L,P)
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_seq)  # shape (L,P)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    # Apply state emissions
    conj_sym_fctr = 2 if conj_sym else 1
    ys = conj_sym_fctr * jax.vmap(lambda x: (C_tilde @ x).real)(xs) + D * input_seq
    
    return xs[-1], ys


class S5SSM(nn.Module):
    Lambda_re_init: jax.Array
    Lambda_im_init: jax.Array
    V: jax.Array
    Vinv: jax.Array

    H: int
    P: int
    C_init: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    activation: str = "gelu"

    """ The S5 SSM

    The state space model is parameterized in complex modal coordinates.
    Consider the following real-valued state space model with state transtion matrix A,
    state input matrix B, state emission matrix C, and input feedthrough matrix D:
        z[i] = A z[i-1] + B u[i]
        y[i] = C z[i] + D u[i]
    The modal representation of this system uses the diagonaled state transition matrix
    Lambda, where A = Vinv @ Lambda @ V, and transformed state x[i] = V @ z[i].
    Then, the above system is equivalently expressed as
        x[i] = Lambda x[i-1] + B_tilde u[i]
        y[i] = C_tilde x[i] + D u[i]
    where B_tilde = Vinv @ B and C_tilde = C @ V.

    Args:
        Lambda_re_init (float32): Real part of init diag state matrix  (P_,)
        Lambda_im_init (float32): Imag part of init diag state matrix  (P_,)
        V           (complex64): Eigenvectors used for init           (P,P_)
        Vinv        (complex64): Inverse eigenvectors used for init   (P_,P)
        H           (int32):     Feature dimension
        P           (int32):     State dimension.
            Denoted as P_ in docstrings. If conj_sym, P_ = P//2, where P is true state dimension
            (referred to as 'local_P' in setup).
        C_init      (string):    Method for initializing emissions matrix C. Options:
            - 'trunc_standard_normal': Sample from truncated standard normal,
                                       then apply modal transform, i.e. C_tilde = C @ V
            - 'lecun_normal': Sample from Lecun_normal, then apply modal transform, i.e. C_tilde = C @ V.
            - 'complex_normal': Directly sample a complex valued output matrix from standard normal.
                                No further transformation applied, i.e. C_tilde = C.
        dt_min:      (float32): minimum value to draw timescale values from when 
                                initializing log_step
        dt_max:      (float32): maximum value to draw timescale values from when 
                                initializing log_step
        conj_sym    (bool):    Whether conjugate symmetry is enforced
        clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                constrain real part of eigenvalues to be negative. 
                                True recommended for autoregressive task/unbounded sequence lengths
                                Discussed in https://arxiv.org/pdf/2206.11893.pdf.
        activation   (str):    type of activation to apply to SSM outputs

    nn.Module params:
        Lambda_re: Array[float32], (P_,). Real part of diagonal transition matrix.
        Lambda_im: Array[float32], (P_,). Imaginary part of diagonal transition matrix.
        B: Array[float32], (P_, H, 2).
            Real and imaginary part of input matrix, in modal_coordinates.
        C: Array[float32], (H, state_dim, 2).
            Real and imaginary part of state emissions matrix, in modal_coordinates.
        D: Array[float32], (H,). Diagonal feedthrough matrix.
        log_step: Array[float32], (H,1). Per-feature timescale discretization value.
    where P_ = P//2 if conj_sym else P_ = P
        
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5, dtype="complex")  # dtype="complex": allow jax backend to set correct precision
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            self.C = self.param("C", C_init, (self.H, self.P, 2))

        else:
            self.C = self.param("C",
                                lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                C_shape)

        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = np.exp(self.log_step[:, 0])

        # Define activations
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.H)
            self.out2 = nn.Dense(self.H)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.H)

        # Discretize
        self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)

    def __call__(self, input_sequence, training=True):
        """Compute output sequence given an input sequence using a parallel scan.

        Args:
            input_sequence: Array[float32], shape (bsz, n_heads, H, n_seq_blocks, L)
                The 5-dimensional is due to the shape imposed by the original Hyena implementation.
                S5 assumes n_heads and n_seq_blocks are singleton dimensions.
        
        Returns:
            output_sequence Array[float32](bsz, n_heads, H, n_seq_blocks, L)
        """

        input_sequence = input_sequence[:, 0, :, 0]  # Remove singleton dimensions
        input_sequence = input_sequence.transpose(0, 2, 1)  # Now, (bsz, L, H)

        # Apply input sequence to SSM using parallel scan. vmap over bsz axis.
        # Returns ys: shape (bsz, L, H)
        _, ys = jax.vmap(
            apply_ssm, in_axes=(0, None, None, None, None, None)
        )(input_sequence, self.Lambda_bar, self.B_bar, self.C_tilde, self.D, self.conj_sym)

        if self.activation in ["full_glu"]:
            ys = nn.activation.gelu(ys, approximate=False)
            ys = self.out1(ys) * jax.nn.sigmoid(self.out2(ys))
        elif self.activation in ["half_glu1"]:
            ys = nn.activation.gelu(ys, approximate=False)
            ys = ys * jax.nn.sigmoid(self.out2(ys))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.activation.gelu(ys, approximate=False)
            ys = ys * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            ys = nn.activation.gelu(ys, approximate=False)
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))

        output_sequence = np.expand_dims(ys.transpose(0, 2, 1), (1, 3))  # Now, (bsz,1,H,1,L)

        return output_sequence

    def step(self, state, inpt, training=False):
        """Compute single step of S5 DSSM given an input.

        Args:
            state: Array[complex64], shape (bsz, state_dim)
                State at last timestep.
            inpt: Array[float32], shape (bsz, n_heads, H, n_blocks)
                The singleton sequence length dimension has been removed.
                Otherwise, the (now 4-dim) shape is due to original Hyena implementation.
                S5 assumes n_heads and n_seq_blocks are singleton dimensions.
                Using argument name `inpt` to avoid clash with built-in `input` function.
        
        Returns:
            new_state: Array[complex64], shape (bsz, state_dim)
            output: Array[float32], shape (bsz, n_heads, H, n_blocks)
                Return in (new_state, output) order to be consistent with
                output signature (carry, y) of scan functions, e.g. jax.lax.scan or nn.scan
        """

        # Remove singleton axes
        inpt = inpt[:, 0, :, 0]

        # Apply single SSM step; vmap over bsz axis
        # Recall that we define our state-space model (in the real standard case) as
        #   x[i] = A x[i-1] + B u[i]
        #   y[i] = C x[i] + D u[i]
        # so make sure to compute output y with the new state.
        fctr = 2 if self.conj_sym else 1
        new_state = jax.vmap(lambda x, u: self.Lambda_bar * x + self.B_bar @ u)(state, inpt)
        y = jax.vmap(lambda x, u: fctr*(self.C_tilde @ x).real + self.D * u)(new_state, inpt)

        # Apply non-linear function
        if self.activation in ["full_glu"]:
            y = nn.activation.gelu(y, approximate=False)
            y = self.out1(y) * jax.nn.sigmoid(self.out2(y))
        elif self.activation in ["half_glu1"]:
            y = nn.activation.gelu(y, approximate=False)
            y = y * jax.nn.sigmoid(self.out2(y))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.activation.gelu(y, approximate=False)
            y = y * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            y = nn.activation.gelu(y, approximate=False)
        else:
            raise NotImplementedError(
                "Activation: {} not implemented".format(self.activation))

        output = y[:,None,:,None]   # Now, (bsz, 1, H, 1)
        
        return new_state, output

def init_S5SSM(d_model, ssm_size, blocks, ssm_args):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""

    block_size = int(ssm_size / blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)

    if ssm_args['conj_sym']:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    return S5SSM(Lambda.real,
                 Lambda.imag,
                 V,
                 Vinv,
                 H=d_model,
                 P=ssm_size,
                 **ssm_args)


class S5Operator(nn.Module):
    d_model: int
    n_layer: int
    l_max: int
    ssm_size: int = 64
    ssm_blocks: int = 1
    order: int = 2
    num_heads: int = 1
    inner_factor: int = 1
    num_blocks: int = 1
    fused_bias_fc: bool = False
    outer_mixing: bool = False
    drop_rate: float = 0.0
    filter_dropout: float = 0.0
    filter_cls: str = 'None'
    post_order_ffn: bool = False
    jit_filter: bool = False
    short_filter_order: int = 3
    activation_type: str = "id"
    return_state: bool = False
    filter_args: dict = None

    def setup(self):
        r"""
        Hyena operator described in the paper https://arxiv.org/pdf/2302.10866.pdf

        Args:
            d_model (int): Dimension of the input and output embeddings (width of the layer)
            n_layer (int): # of model layers, (used for special scaled init)
            l_max: (int): Maximum input sequence length. Defaults to None
            ssm_size: (int): Size of the ssm
            ssm_blocks: (int): Number of initial blocks to use when initialzing SSM state matrix
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            num_heads: (int): Number of heads. Defaults to 1
            inner_factor: (int): Width multiplier. Defaults to 1
            num_blocks: (int): Number of blocks in sequence length. Defaults to 1
            fused_bias_fc: (bool): Whether to use fused bias FC. Defaults to False
            drop_rate: (float): Dropout probability. Defaults to 0.0
            filter_dropout: (float): Dropout probability for the filter. Defaults to 0.0
            post_order_ffn: (bool): Apply a dense layer between steps of the recurrence. Defaults to False
            jit_filter: (bool): Whether JIT the implicit filter function. Defaults to False
            short_filter_order: (int): Length of the explicit input convolutional filter. Defaults to 3
            activation_type: (str): type of act between kernel output and FF (default identity)
            return_state: (bool): whether to return a state
        """

        assert self.d_model % self.num_heads == 0, f'Model dimension {self.d_model} must be divisible by num heads {self.num_heads}'
        assert self.l_max % self.num_blocks == 0, f'Maximum signal length {self.l_max} must be divisible by block dimension {self.num_blocks}'
        block_dim = self.l_max // self.num_blocks
        self.head_dim = self.d_model // self.num_heads

        self.activation = Activation(self.activation_type)
        self.dropout = partial(nn.Dropout, self.drop_rate)
        self.setup_projections(self.fused_bias_fc, self.inner_factor)
        self.setup_filters(self.filter_cls, self.filter_args)

    def setup_projections(self, fused_bias_fc, inner_factor, initializer_range=0.02):
        "Initializes input and output projections (over the width dimension)"

        # if fused_bias_fc and FusedDense is None:
        if fused_bias_fc:
            raise ImportError('fused_dense is not installed')
        if not fused_bias_fc:
            linear_cls = nn.Dense

        out_kernel_init = flax_normal(stddev=initializer_range / math.sqrt(2 * self.n_layer))
        self.out_proj = linear_cls(self.d_model, kernel_init=out_kernel_init)
        self.in_proj = linear_cls((self.order + 1) * self.d_model)
        if self.post_order_ffn:
            self.ord_proj_w = self.param("ord_proj_w",
                                         normal(stddev=1/math.sqrt(self.head_dim)),
                                         (self.order, self.num_heads, self.num_heads))

    def setup_filters(self, filter_cls, filter_args):
        "Initializes the explicit and implicit filters"
        assert self.order >= 2, f'Order must be at least 2, (got {self.order})'
        total_width = self.d_model * self.inner_factor * (self.order + 1)

        self.short_filter = nn.Conv(total_width,
                                    [self.short_filter_order],
                                    feature_group_count=total_width,
                                    padding=self.short_filter_order - 1)

        if self.filter_cls == 'hyena_S5':
            # print('Using S5 for filters')
            self.filter_fn = [init_S5SSM(self.d_model, self.ssm_size, self.ssm_blocks, filter_args) for _ in range(self.order-1)]
        else:
            raise NotImplementedError("filter {} not implemented".format(self.filter_cls))

    @nn.compact
    def __call__(self, u, training):
        l = u.shape[-2]
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)
        # u = rearrange(u, 'b l d -> b d l')

        # note u is still 'b l d'
        uc = self.short_filter(u)[:, :l_filter]
        # uc is 'b l d'
        uc = rearrange(uc, 'b l d -> b d l')
        uc = rearrange(uc, 'b (ho v) (z l) -> b ho v z l',
                       z=self.num_blocks,
                       ho=self.num_heads,
                       v=self.head_dim * (self.order + 1)
                       )

        width = uc.shape[2]
        split_width = int(width // self.d_model)
        *x, v = np.split(uc, split_width, axis=2)

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                raise NotImplementedError("outer mixing not implemented for hyena_S5 yet")
            else:
                v = self.dropout(deterministic=not training)(v * x_i)

            v = self.filter_fn[o](v)

            if self.post_order_ffn:
                w = self.ord_proj_w[o]
                v = mul_sum(
                    rearrange(w, 'h1 h2 -> 1 h1 h2 1 1 1'), rearrange(v, 'b h v z l -> b h 1 v z l')
                )

        y = self.activation(rearrange(v * x[0], 'b h v z l -> b (z l) (h v)', z=self.num_blocks, h=self.num_heads))
        y = self.out_proj(y)

        if self.return_state:
            return y, None
        return y

    @property
    def d_output(self):
        return self.d_model

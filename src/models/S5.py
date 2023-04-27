import jax
import jax.numpy as np
from einops import rearrange, repeat
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from jax.nn.initializers import lecun_normal, normal
from jax.scipy.linalg import block_diag
import math
from functools import partial

from .hyena import Activation, mul_sum
from .SSM_init import init_CV, init_VinvB,\
    make_DPLR_HiPPO, init_log_steps, trunc_standard_normal


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
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


def apply_ssm_with_feedthrough(input_sequence, Lambda_bar, B_bar, C_tilde, D, conj_sym):
    ys = apply_ssm(Lambda_bar,
                   B_bar,
                   C_tilde,
                   input_sequence,
                   conj_sym)

    # Add feedthrough matrix output Du;
    Du = jax.vmap(lambda u: D * u)(input_sequence)
    output_sequence = ys + Du
    return output_sequence


class S5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    C_init: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    activation: str = "gelu"

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
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
            C_init = normal(stddev=0.5 ** 0.5)
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
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (bsz, num_heads, H, num_blocks, L)
                                       where for now num_heads and num_blocks is 1
        Returns:
            output sequence (float32): (bsz, num_heads, H, num_blocks, L)
        """

        input_sequence = input_sequence[:, 0, :, 0]
        input_sequence = input_sequence.transpose(0, 2, 1)
        # input sequence is now bsz, L, H

        ys = jax.vmap(apply_ssm_with_feedthrough,
                      in_axes=(0, None, None, None, None, None)
                      )(input_sequence,
                        self.Lambda_bar,
                        self.B_bar,
                        self.C_tilde,
                        self.D,
                        self.conj_sym)  # ys is bsz, L, H

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
        # ys is bsz, L, H
        output_sequence = np.expand_dims(ys.transpose(0, 2, 1), (1, 3))
        # output sequence is bsz, 1, H, 1, L

        return output_sequence


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

    # print("Lambda.shape={}".format(Lambda.shape))
    # print("V.shape={}".format(V.shape))
    # print("Vinv.shape={}".format(Vinv.shape))

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

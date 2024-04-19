"""
Quantized version of the S5 layer implementation from lindermanlab/S5.

Using the `aqt` JAX library for quantization.

TODOs:
- test this
"""
from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal
from .ssm import discretize_bilinear, discretize_zoh
from utils.quantization import q_dot_maybe, q_had_maybe

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class QuantizationConfig:
    """Quantization configuration for S5.

    Attributes:
        a_precision: integer precision for A matrix operations.
        b_precision: integer precision for B matrix operations.
        c_precision: integer precision for C matrix operations.
        d_precision: integer precision for D matrix operations.
    """
    a_precision: Optional[int]
    b_precision: Optional[int]
    c_precision: Optional[int]
    d_precision: Optional[int]


@dataclass
class QuantizedOperations:
    """(Possibly quantized) operations for S5.

    Attributes:
        a_dot: (possibly quantized) dot product operation for A matrix.
        a_had: (possibly quantized) hadamard product operation for A matrix.
        b_dot: (possibly quantized) dot product operation for B matrix.
        c_dot: (possibly quantized) dot product operation for C matrix.
        d_had: (possibly quantized) hadamard product operation for D matrix.
    """
    a_dot: Callable
    a_had: Callable
    b_dot: Callable
    c_dot: Callable
    d_had: Callable

    def __init__(self, q_config: QuantizationConfig):
        self.a_dot = q_dot_maybe(q_config.a_precision)
        self.a_had = q_had_maybe(q_config.a_precision)
        self.b_dot = q_dot_maybe(q_config.b_precision)
        self.c_dot = q_dot_maybe(q_config.c_precision)
        self.d_had = q_had_maybe(q_config.d_precision)


# Parallel scan operations
@jax.vmap
def quant_binary_operator(q_i, q_j, qdot_fn, qhad_fn):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j

    def qadd_fn(x, y):
        """Add two quantized vectors, using the quantized dot function."""
        return qdot_fn(np.array([x, y]), np.ones((2,)))

    # return A_j * A_i, A_j * b_i + b_j
    return qhad_fn(A_j, A_i), qadd_fn(qhad_fn(A_j, b_i), b_j)


def build_apply_ssm(q_ops: QuantizedOperations) -> Callable:

    q_bin_op = partial(quant_binary_operator, qdot_fn=q_ops.a_dot, qhad_fn=q_ops.a_had)

    def _apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
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
        Bu_elements = jax.vmap(lambda u: q_ops.b_dot(B_bar, u))(input_sequence)

        _, xs = jax.lax.associative_scan(q_bin_op, (Lambda_elements, Bu_elements))

        if bidirectional:
            _, xs2 = jax.lax.associative_scan(q_bin_op,
                                              (Lambda_elements, Bu_elements),
                                              reverse=True)
            xs = np.concatenate((xs, xs2), axis=-1)

        if conj_sym:
            return jax.vmap(lambda x: 2*q_ops.c_dot(C_tilde, x).real)(xs)
        else:
            return jax.vmap(lambda x: q_ops.c_dot(C_tilde, x).real)(xs)

    return jax.jit(_apply_ssm)


class S5SSM(nn.Module):
    Lambda_re_init: np.DeviceArray
    Lambda_im_init: np.DeviceArray
    V: np.DeviceArray
    Vinv: np.DeviceArray

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0
    q_config: QuantizationConfig

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
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative.
                                   True recommended for autoregressive task/unbounded sequence
                                   lengths. Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g.
                                    after training on a different resolution for the speech
                                    commands benchmark
            q_config:    (QuantizationConfig): Configuration for quantization.
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        self.q_ops = QuantizedOperations(self.q_config)
        self.apply_ssm = build_apply_ssm(self.q_ops)

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
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

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
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization}")

    def __call__(self, input_sequence):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        ys = self.apply_ssm(self.Lambda_bar,
                            self.B_bar,
                            self.C_tilde,
                            input_sequence,
                            self.conj_sym,
                            self.bidirectional)

        # Add feedthrough matrix output Du;
        # self.D * u can be replaced with the quant vector product einsum now.
        Du = jax.vmap(lambda u: self.q_ops.d_had(self.D, u))(input_sequence)
        return ys + Du  # TODO: make sure this is also quantized


def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)

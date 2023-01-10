from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal

from .ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


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


def apply_ssm(Lambda_elements, Bu_elements, C_tilde, conj_sym, bidirectional, Lambda_elements_bwd=None, Bu_elements_bwd=None):
    """ Compute the LxH output of discretized SSM given an LxH input.

        CHANGELOG:  04-JAN-2023:  Bringing this function in to complians with
        variable interval applications.  Instead, the Lambda and B elements
        must be pre-computed for both the forward pass and any backwards pass.
        Failure to provide the (exactly) correct arguments will throw an
        error.

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

    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

    if bidirectional:
        assert (Lambda_elements_bwd is not None) and (Bu_elements_bwd is not None), "Must provide bwd kernels."
        _, xs2 = jax.lax.associative_scan(binary_operator, (Lambda_elements_bwd, Bu_elements_bwd), reverse=True)
        xs = np.concatenate((xs, xs2), axis=-1)
    else:
        assert (Lambda_elements_bwd is None) and (Bu_elements_bwd is None), "Cannot provide bwd kernels."

    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return jax.vmap(lambda x: (C_tilde @ x).real)(xs)


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
    variable_observation_interval: bool
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

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
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
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
            self.discretize_fn = discretize_zoh
        elif self.discretization in ["bilinear"]:
            self.discretize_fn = discretize_bilinear
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

        if not self.variable_observation_interval:
            # Apply the discretize ahead of time.
            self.Lambda_bar, self.B_bar = self.discretize_fn(self.Lambda, B_tilde, step)
        else:
            # If we have variable observation interval, then we will need to discretize on-the-fly.
            self.Lambda_bar, self.B_bar = None, None

    def __call__(self, input_sequence, integration_timesteps=None):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
        Returns:
            output sequence (float32): (L, H)
        """
        # These will be over-written in the case that we have a bi-directional model.
        Lambda_bar_elements_bwd = None
        Bu_bar_elements_bwd = None

        # If we have variable observation intervals, then we need to compute the variables on the fly.
        if not self.variable_observation_interval:
            # Fixed interval observations (or the interval is being dealt with elsewhere).
            assert (self.Lambda_bar is not None) and (self.B_bar is not None), "Must be pre-computed."
            Lambda_bar_elements = self.Lambda_bar * np.ones((input_sequence.shape[0],
                                                             self.Lambda_bar.shape[0]))
            Bu_bar_elements = jax.vmap(lambda u: self.B_bar @ u)(input_sequence)

            # For fixed-interval observations, the reverse terms are the same as the forward terms.
            if self.bidirectional:
                Lambda_bar_elements_bwd = Lambda_bar_elements
                Bu_bar_elements_bwd = Bu_bar_elements

        else:
            assert (self.Lambda_bar is None) and (self.B_bar is None), "Cannot pre-compute these.  How are these not `None`..."

            # TODO - Testing this implementation.  Including this entire function, in fact...
            @jax.vmap
            def _do_vmapped_discretize(_timestep):
                print('\nWarning: Discretizing on-the-fly...\n')
                B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
                step = self.step_rescale * np.exp(self.log_step[:, 0])
                Lambda_bar, B_bar = self.discretize_fn(self.Lambda, B_tilde, step * _timestep)
                return Lambda_bar, B_bar

            # Discretize forward pass.
            fwd_timesteps = np.expand_dims(np.concatenate((np.asarray((1,)), integration_timesteps)), -1)
            Lambda_bar_elements, B_bar_elements = _do_vmapped_discretize(fwd_timesteps)
            Bu_bar_elements = jax.vmap(lambda u, b: b @ u)(input_sequence, B_bar_elements)

            if self.bidirectional:
                bwd_timesteps = np.expand_dims(np.concatenate((integration_timesteps, np.asarray((1,)))), -1)
                Lambda_bar_elements_bwd, B_bar_elements_bwd = _do_vmapped_discretize(bwd_timesteps)
                Bu_bar_elements_bwd = jax.vmap(lambda u, b: b @ u)(input_sequence, B_bar_elements_bwd)

        ys = apply_ssm(Lambda_bar_elements,
                       Bu_bar_elements,
                       self.C_tilde,
                       self.conj_sym,
                       self.bidirectional,
                       Lambda_elements_bwd=Lambda_bar_elements_bwd,
                       Bu_elements_bwd=Bu_bar_elements_bwd, )

        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


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
               variable_observation_interval,
               conj_sym,
               clip_eigs,
               bidirectional,
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
                   variable_observation_interval=variable_observation_interval,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)

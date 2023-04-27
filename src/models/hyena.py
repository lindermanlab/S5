import jax.numpy as np
from einops import rearrange, repeat
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
import math
from jax.nn.initializers import normal
from functools import partial


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        # return nn.Identity()
        return lambda x: x
    elif activation == 'tanh':
        return nn.tanh
    elif activation == 'relu':
        return nn.relu
    elif activation == 'gelu':
        return nn.gelu
    elif activation in ['swish', 'silu']:
        return nn.silu
    elif activation == 'glu':
        return partial(nn.glu, axis=dim)
    elif activation == 'sigmoid':
        return nn.sigmoid
    elif activation == 'softplus':
        return nn.softplus
    # elif activation in ['sqrelu', 'relu2']:
    #     return SquaredReLU()
    # elif activation == 'laplace':
    #     return Laplace()
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))


def fftconv_ref(u, k, D, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = np.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = np.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = np.fft.rfft(u, n=fft_size)

    # if len(u.shape) > 3: k_f = k_f.unsqueeze(1)
    if len(u.shape) > 3:
        k_f = np.expand_dims(k_f, 1)

    y = np.fft.irfft(u_f * k_f, n=fft_size, norm='forward')[..., :seqlen]

    out = y + u * np.expand_dims(D, -1)
    if gelu:
        out = nn.activation.gelu(out, approximate=False)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, 'b H -> b H 1'))
    else:
        return out


def mul_sum(q, y):
    return np.sum(q * y, axis=1)


class Sin(nn.Module):
    dim: int
    w: float = 10
    train_freq: bool = True

    def setup(self):
        if self.train_freq:
            self.freq = self.param("freq", lambda rng, shape: self.w * np.ones((1, self.dim)), (None,))
        else:
            self.freq = self.w * np.ones((1, self.dim))

    def __call__(self, x):
        return np.sin(self.freq * x)


class PositionalEmbedding(nn.Module):
    emb_dim: int
    seq_len: int
    lr_pos_emb: float = 1e-5

    def setup(self):
        """Complex exponential positional embeddings for Hyena filters."""
        # The time embedding fed to the filters is normalized so that t_f = 1
        # self.t = self.param("t",
        #                     lambda rng, shape: np.linspace(0, 1, self.seq_len)[None, :, None],
        #                     (None,) ) # 1, L, 1

        self.t = np.linspace(0, 1, self.seq_len)[None, :, None]

        def init_z():
            if self.emb_dim > 1:
                bands = (self.emb_dim - 1) // 2
                # To compute the right embeddings we use the "proper" linspace
            t_rescaled = np.linspace(0, self.seq_len - 1, self.seq_len)[None, :, None]
            w = 2 * math.pi * t_rescaled / self.seq_len  # 1, L, 1

            f = np.linspace(1e-4, bands - 1, bands)[None, None]
            z = np.exp(-1j * f * w)
            z = np.concatenate([self.t, z.real, z.imag], axis=-1)
            return z

        self.z = init_z()

    def __call__(self, L):
        return self.z[:, :L], self.t[:, :L]


class ExponentialModulation(nn.Module):
    d_model: int
    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    modulation_lr: float = 0.0
    modulate: bool = True
    shift: float = 0.0

    def setup(self):
        def init_deltas():
            max_decay = math.log(self.target) / self.fast_decay_pct
            min_decay = math.log(self.target) / self.slow_decay_pct
            deltas = np.linspace(min_decay, max_decay, self.d_model)[None, None]
            return deltas

        self.deltas = init_deltas()

    def __call__(self, t, x):
        if self.modulate:
            decay = np.exp(-t * np.abs(self.deltas))
            x = x * (decay + self.shift)
        return x


class HyenaFilter(nn.Module):
    d_model: int
    emb_dim: int = 3  # dim of input to MLP, augments with positional encoding
    order: int = 16  # width of the implicit MLP
    fused_fft_conv: bool = False
    seq_len: int = 1024
    lr: float = 1e-3
    lr_pos_emb: float = 1e-5
    drop_rate: float = 0.0
    w: int = 1  # frequency of periodic activations
    wd: int = 0  # weight decay of kernel parameters
    use_bias: bool = True
    num_inner_mlps: int = 2
    normalized: bool = False

    def setup(self):
        """
        Implicit long filter with modulation.

        Args:
            d_model: number of channels in the input
            emb_dim: dimension of the positional encoding (`emb_dim` - 1) // 2 is the number of bands
            order: width of the FFN
            num_inner_mlps: number of inner linear layers inside filter MLP

        Note:
            filter_dropout is not implemented
        """

        self.bias = self.param("bias", normal(stddev=1.0), (self.d_model,))

        act = Sin(dim=self.order, w=self.w)
        assert self.emb_dim % 2 != 0 and self.emb_dim >= 3, "emb_dim must be odd and greater or equal to 3 (time, sine and cosine)"

        self.pos_emb = PositionalEmbedding(self.emb_dim, self.seq_len, self.lr_pos_emb)

        # uses a variable number of inner linear layers
        implicit_filter_list = [nn.Dense(self.order), act]
        for i in range(self.num_inner_mlps):
            implicit_filter_list.append(nn.Dense(self.order))
            implicit_filter_list.append(act)

        implicit_filter_list.append(nn.Dense(self.d_model, use_bias=False))
        self.implicit_filter = nn.Sequential(implicit_filter_list, name='implicit_filter')
        self.modulation = ExponentialModulation(self.d_model)

    def filter(self, L):
        z, t = self.pos_emb(L)
        h = self.implicit_filter(z)
        h = self.modulation(t, h)

        if self.normalized:
            h = h / np.linalg.norm(h, ord=1, axis=-1,  keepdims=True)

        return h

    def __call__(self, x, L, k=None, bias=None):
        if k is None:
            k = self.filter(L)

        # Ensure compatibility with filters that return a tuple
        k = k[0] if type(k) is tuple else k
        if bias is None:
            bias = self.bias
        bias = bias if self.use_bias else 0 * bias

        if self.fused_fft_conv:
            pass
        else:
            y = fftconv_ref(x, k, bias, dropout_mask=None, gelu=False)

        return y


class HyenaOperator(nn.Module):
    d_model: int
    n_layer: int
    l_max: int
    order: int = 2
    filter_order: int = 64
    num_heads: int = 1
    inner_factor: int = 1
    num_blocks: int = 1
    fused_bias_fc: bool = False
    outer_mixing: bool = False
    drop_rate: float = 0.0
    filter_dropout: float = 0.0
    filter_cls: str = 'hyena-filter'
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
            order: (int): Depth of the Hyena recurrence. Defaults to 2
            filter_order: (int): Width of the FFN parametrizing the implicit filter. Defaults to 64
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

        if self.filter_cls == 'hyena-filter':
            # print('using Hyena-filter')
            filter_cls = HyenaFilter
        else:
            raise NotImplementedError("filter {} not implemented".format(self.filter_cls))

        self.filter_fn = filter_cls(
            self.head_dim * self.inner_factor * (self.order - 1),
            order=self.filter_order,
            seq_len=self.l_max,
            drop_rate=self.filter_dropout,
            **filter_args
        )
        # if self.jit_filter: self.filter_fn = torch.jit.script(self.filter_fn, self.L)

    def recurrence(self, u, state):
        "Fast inference mode via distilled recurrence"
        raise NotImplementedError("Working on it!")

    @nn.compact
    def __call__(self, u, training):
        l = u.shape[-2]
        l_filter = min(l, self.l_max)
        u = self.in_proj(u)

        # note u is still 'b l d'
        uc = self.short_filter(u)[:, :l_filter]
        # uc is 'b l d'
        uc = rearrange(uc, 'b l d -> b d l')
        uc = rearrange(uc, 'b (ho v) (z l) -> b ho v z l',
                       z=self.num_blocks,
                       ho=self.num_heads,
                       v=self.head_dim * (self.order + 1)
                       )

        # Note jax.numpy.split has diff convention from torch.split()
        width = uc.shape[2]
        split_width = int(width // self.d_model)
        *x, v = np.split(uc, split_width, axis=2)
        k = self.filter_fn.filter(l_filter)

        # `c` is always 1 by default
        k = rearrange(k, 'c l (v o) -> c o v l', v=self.head_dim, o=self.order - 1)[0]
        bias = rearrange(self.filter_fn.bias, '(v o) -> o v', v=self.head_dim, o=self.order - 1)

        for o, x_i in enumerate(reversed(x[1:])):
            if self.outer_mixing:
                v = rearrange(v, 'b h v z l -> b h 1 v z l')
                v = self.dropout(deterministic=not training)(
                    v * rearrange(x_i, 'b h v z l -> b h v 1 z l')
                )
                v = np.sum(v, axis=2)
            else:
                v = self.dropout(deterministic=not training)(v * x_i)

            # the bias term is broadcasted. Last dimension (l) is handled by fftconv
            v = self.filter_fn(v, l_filter, k=k[o], bias=bias[o, None, :, None])

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

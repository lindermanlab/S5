import jax.numpy as np
from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
from functools import partial
import math

from .S5 import S5Operator
from .hyena import HyenaOperator
from .utils import StochasticDepth, Identity
from collections import namedtuple


class GPT2Embeddings(nn.Module):
    embed_dim: int
    vocab_size: int
    max_position_embeddings: int
    padding_idx: int = None  # TODO: not currently used
    word_embed_proj_dim: int = None

    def setup(self):
        """
            If max_position_embeddings <= 0, there's no position embeddings
            If word_embe_proj_dim is not None (e.g., OPT-350m), we embed to that dimension
                the project up to embed_dim
        """
        if self.word_embed_proj_dim is None:
            self.word_embeddings = nn.Embed(self.vocab_size, self.embed_dim)
            # self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx,
            #                                     **factory_kwargs)
            self.project_in = None
        else:
            self.word_embeddings = nn.Embed(self.vocab_size, self.word_embed_proj_dim)
            # self.word_embeddings = nn.Embedding(vocab_size, word_embed_proj_dim,
            #                                     padding_idx=padding_idx, **factory_kwargs)
            self.project_in = nn.Dense(self.embed_dim, use_bias=False)

        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embed(self.max_position_embeddings, self.embed_dim)

    def attend(self, input):
        """Use for weight sharing to produce output logits of model"""
        return self.word_embeddings.attend(input)

    def __call__(self, input_ids, position_ids=None):
        """
            input_ids: (batch, seqlen)
            position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape
        embeddings = self.word_embeddings(input_ids)

        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = np.arange(seqlen, dtype=np.int64)
            # if position_ids is None:
            #     position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings


class Mlp(nn.Module):
    in_features: int
    n_layer: int  # This is the number of model layers used for init
    hidden_features: int = None
    out_features: int = None
    activation: nn.module = nn.gelu
    return_residual: bool = False
    initializer_range: float = 0.02

    def setup(self):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/mlp.py
        """
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        self.fc1 = nn.Dense(hidden_features)

        out_kernel_init = flax_normal(stddev=self.initializer_range / math.sqrt(2 * self.n_layer))
        self.fc2 = nn.Dense(out_features, kernel_init=out_kernel_init)

    def __call__(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)


class Block(nn.Module):
    dim: int
    n_layer: int  # This is number of model layers used for init
    mixer_cls: nn.Module = None
    mlp_cls: nn.Module = None
    norm_cls: nn.Module = nn.LayerNorm
    dropout_cls: nn.Module = nn.Dropout
    prenorm: bool = True
    resid_dropout1: float = 0.0
    resid_dropout2: float = 0.0
    drop_path1_rate: float = 0.0
    drop_path2_rate: float = 0.0
    return_residual: bool = False
    # residual_in_fp32: bool = False

    def setup(self):
        """
        From https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
        For prenorm=True, this Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Dropout -> Add -> LN -> MHA -> Dropout -> Add -> LN -> MLP, returning both
        the hidden_states (output of the MLP) and the residual.
        This is for performance reasons, as we can fuse the dropout, add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        For prenorm=False, this Block has the same structure as a regular postnorm Transformer
        block: MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        # if self.residual_in_fp32:
        #     assert self.prenorm, 'residual_in_fp32 is only compatible with prenorm=True'
        if self.mixer_cls is None:
            raise NotImplementedError("MHA not implemented")
            # mixer_cls = partial(MHA, num_heads=dim // 64)
        if self.mlp_cls is None:
            self.mlp_cls = partial(Mlp, hidden_features=4 * self.dim)
        # self.mixer = self.mixer_cls(self.dim)
        self.mixer = self.mixer_cls
        self.dropout1 = partial(self.dropout_cls, self.resid_dropout1)
        self.drop_path1 = StochasticDepth(p=self.drop_path1_rate, mode='row')
        self.norm1 = self.norm_cls()
        self.mlp = self.mlp_cls(self.dim, self.n_layer)
        if not isinstance(self.mlp, Identity):
            # TODO: Check that this path works
            self.dropout2 = partial(self.dropout_cls, self.resid_dropout2)
            self.drop_path2 = StochasticDepth(p=self.drop_path2_rate, mode='row')
            self.norm2 = self.norm_cls()

    @nn.compact
    def __call__(self, hidden_states, training, residual=None,
                 mixer_subset=None, mixer_kwargs=None):
        r"""Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            training: bool to control dropout
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            dropped = self.drop_path1(self.dropout1(deterministic=not training)(hidden_states), training)
            residual = (dropped + residual) if residual is not None else dropped
            hidden_states = self.norm1(residual)

            if mixer_kwargs is None:
                mixer_kwargs = {}
            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            hidden_states = self.mixer(hidden_states, training, **mixer_kwargs)
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]
            if not isinstance(self.mlp, Identity):
                dropped = self.drop_path2(self.dropout2(deterministic=not training)(hidden_states), training)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.norm2(residual)

                hidden_states = self.mlp(hidden_states)
            return hidden_states, residual
        else:
            assert residual is None
            mixer_out = self.mixer(
                hidden_states, **(mixer_kwargs if mixer_kwargs is not None else {})
            )
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_states = mixer_out

            hidden_states = self.norm1(self.drop_path1(self.dropout1(deterministic=not training)(mixer_out), training)
                                       + hidden_states)

            if not isinstance(self.mlp, Identity):
                mlp_out = self.mlp(hidden_states)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_states = mlp_out

                hidden_states = self.norm2(self.drop_path2(self.dropout2(deterministic=not training)(mlp_out), training)
                                           + hidden_states)

            return hidden_states


def create_mixer_cls(layer=None, d_model=None, n_layer=None, l_max=None, layer_kwargs=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None):
    if attn_layer_idx is not None and layer_idx in attn_layer_idx:
        raise NotImplementedError("MHA not implemented")
        # causal = True if attn_cfg is None else attn_cfg.pop('causal', True)
        #
        # mha_cls = MHA
        #
        # mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx,
        #                     **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    else:
        if layer == "hyena":
            # mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **factory_kwargs)
            mixer_cls = HyenaOperator(d_model, n_layer, l_max, **layer_kwargs)

        elif layer == "S5_operator":
            mixer_cls = S5Operator(d_model, n_layer, l_max, **layer_kwargs)

    return mixer_cls


def create_mlp_cls(d_model, d_inner=None):
    inner_dim = d_inner if d_inner is not None else 4 * d_model

    mlp_cls = partial(Mlp, hidden_features=inner_dim,
                      activation=partial(nn.gelu, approximate=True))

    return mlp_cls


def create_block(d_model, n_layer, l_max=None, layer_kwargs=None, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0,
                 layer_idx=None):
    mixer_cls = create_mixer_cls(layer=layer, d_model=d_model, n_layer=n_layer, l_max=l_max, layer_kwargs=layer_kwargs,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg, layer_idx=layer_idx)
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner)
    norm_cls = partial(nn.LayerNorm, epsilon=layer_norm_epsilon)
    block = Block(d_model, n_layer, mixer_cls, mlp_cls, norm_cls=norm_cls,
                  prenorm=True, resid_dropout1=resid_dropout1, resid_dropout2=resid_dropout2)
    block.layer_idx = layer_idx
    return block


class LMBackbone(nn.Module):
    d_model: int
    n_layer: int
    d_inner: int
    vocab_size: int
    layer: str = None
    l_max: int = None
    layer_kwargs: dict = None
    process_group: int = None
    attn_layer_idx: int = None
    attn_cfg: dict = None
    max_position_embeddings: int = 0
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_cfg: dict = None

    def setup(self):
        self.embeddings = GPT2Embeddings(self.d_model,
                                         self.vocab_size,
                                         self.max_position_embeddings)

        self.layers = [create_block(
            self.d_model, self.n_layer, l_max=self.l_max, layer_kwargs=self.layer_kwargs, d_inner=self.d_inner,
            layer=self.layer, attn_layer_idx=self.attn_layer_idx,
            attn_cfg=self.attn_cfg, layer_norm_epsilon=self.layer_norm_epsilon,
            resid_dropout1=self.embed_dropout if i == 0 else self.resid_dropout,
            resid_dropout2=self.resid_dropout, layer_idx=i) for i in range(self.n_layer)]

        self.drop_f = partial(nn.Dropout, self.resid_dropout)
        self.ln_f = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

    @nn.compact
    def __call__(self, input_ids, training, position_ids=None):
        hidden_states = self.embeddings(input_ids, position_ids=position_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, training, residual)

        dropped = self.drop_f(deterministic=not training)(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual)

        return hidden_states

    def attend(self, input):
        return self.embeddings.attend(input)


class SimpleLMHeadModel(nn.Module):
    d_model: int
    n_layer: int
    d_inner: int
    vocab_size: int
    layer: str = None
    l_max: int = None
    layer_kwargs: dict = None
    attn_layer_idx: int = None
    attn_cfg: dict = None
    max_position_embeddings: int = 0
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_cfg: dict = None
    pad_vocab_size_multiple: int = 1

    def setup(self):
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += self.pad_vocab_size_multiple - (self.vocab_size % self.pad_vocab_size_multiple)

        self.backbone = LMBackbone(
            d_model=self.d_model, n_layer=self.n_layer, d_inner=self.d_inner, vocab_size=self.vocab_size,
            layer=self.layer, l_max=self.l_max, layer_kwargs=self.layer_kwargs, attn_layer_idx=self.attn_layer_idx, attn_cfg=self.attn_cfg,
            max_position_embeddings=self.max_position_embeddings,
            resid_dropout=self.resid_dropout, embed_dropout=self.embed_dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_cfg=self.initializer_cfg
        )

    def __call__(self, input_ids, training=True, position_ids=None, state=None):
        hidden_states = self.backbone(input_ids, training, position_ids=position_ids)
        lm_logits = self.backbone.attend(hidden_states)
        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=lm_logits), None

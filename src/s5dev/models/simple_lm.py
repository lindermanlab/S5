from collections import namedtuple

from functools import partial
import math

from flax import linen as nn
from flax.linen.initializers import normal as flax_normal
import jax
import jax.numpy as np

from s5dev.models.s5 import S5Operator
from s5dev.models.hyena import HyenaOperator
from s5dev.models.utils import StochasticDepth, Identity

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
    """Flash Attention prenorm Transformer block.
    
    For prenorm=True, this Block has a slightly different structure compared to a regular
    prenorm Transformer block. The regular block structure is (Xiong, Yang et al. 2020)
        LN -> <MIXER> -> Dropout -> Add -> LN -> MLP -> Dropout -> Add.
    Here, the prenorm block structure is
        Dropout -> Add -> LN -> <MIXER> -> Dropout -> Add -> LN -> MLP,
    returning both the hidden_states (MLP outputs) and the residual.
    This is for performance reasons, as we can fuse the Dropout, Add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    
    For prenorm=False, this Block has the same structure as a regular postnorm Transformer block:
        <MIXER> -> Dropout -> Add -> LN -> MLP -> Dropout -> Add -> LN.

    Modified from
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/modules/block.py
    
    TODO Review docstring to ensure all applicable. This was copied from the flash-attention module.

    Arguments
        dim: int
        n_layer: int. Number of model layers used for init. TODO Documentation needed
        mixer_cls: nn.Module
            Expected to be instantiated
        mlp_cls: nn.Module
        norm_cls: nn.Module. default: nn.LayerNorm.
        dropout_cls: nn.Module. default: nn.Dropout.
        prenorm: bool. default=True.
            If True, use prenorm block structure; else use postnorm.
            See docstring for more details.
        resid_dropout1: float. default: 0.0
        resid_dropout2: float. default: 0.0
        drop_path1_rate: float. default: 0.0
        drop_path2_rate: float. default: 0.0
        return_residual: bool. default: False
            Only used if prenorm=False. If True, each sub-layer (mixer and mlp) returns
            its residual. This can improve performance because it allows fusing of the
            backward pass of nn.Linear with the residual connection.

    References
        Xiong*, Yang* et al. (ICLR 2020).
        "On Layer Normalization in the Transformer Architecture"
        https://arxiv.org/abs/2002.04745

    """

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

    def setup(self):
        if self.mixer_cls is None:
            raise NotImplementedError("MHA not implemented")
            # mixer_cls = partial(MHA, num_heads=dim // 64)
        self.mixer = self.mixer_cls
        self.dropout1 = partial(self.dropout_cls, self.resid_dropout1)
        self.drop_path1 = StochasticDepth(p=self.drop_path1_rate, mode='row')
        self.norm1 = self.norm_cls()

        # Define MLP
        if self.mlp_cls is None:
            self.mlp_cls = partial(Mlp, hidden_features=4 * self.dim)        
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
            mixer_kwargs: dict. default=None.
                Keyword arguments for mixer class. Only used if mixer_cls=="MHA" (not implemented)
        """
        if self.prenorm:
            # these three lines seem to perform triton `layer_norm_fn`:
            # https://github.com/Dao-AILab/flash-attention/blob/74b0761ff7efc7b90d4e5aeb529c1b2a09a7458c/flash_attn/ops/triton/layer_norm.py
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


    def step(self, mixer_state, hidden_state, residual=None, mixer_subset=None, mixer_kwargs=None):
        """Apply Transformer block to a single input.
        
        Args:
            mixer_state: Array[float32], shape (bsz, ssm_size)
                State of mixer at last time step.
            hidden_state: Array[float32], shape (bsz, d_model).
            residual: Optional. Array[float32], shape (bsz, d_model). default=None.
            mixer_subset: Not used, retained for consistency with __call__.
            mixer_kwargs: dict. default=None.
                Keyword arguments for mixer class. Only used if mixer_cls=="MHA" (not implemented)
        
        Returns:
            new_mixer_state: Array[complex64], shape (bsz, ssm_size)
            output: Array[float32], shape (bsz, d_output)
        """

        if mixer_kwargs is None:
            mixer_kwargs = {}
            
        if self.prenorm:
            dropped = hidden_state  # skipping drop path
            residual = (dropped + residual) if residual is not None else dropped
            hidden_state = self.norm1(residual)

            if mixer_subset is not None:
                mixer_kwargs['mixer_subset'] = mixer_subset
            
            new_mixer_state, hidden_state = self.mixer.step(mixer_state, hidden_state, **mixer_kwargs)
            
            if mixer_subset is not None:
                residual = residual[:, mixer_subset]  # Take a subset before applying the query projectin.

            if not isinstance(self.mlp, Identity):
                dropped = hidden_state  # skipping drop path
                residual = (dropped + residual) if residual is not None else dropped
                hidden_state = self.norm2(residual)

                hidden_state = self.mlp(hidden_state)

            return new_mixer_state, hidden_state, residual

        else:
            assert residual is None
            new_mixer_state, mixer_out = self.mixer.step(mixer_state, hidden_state, **mixer_kwarg)
            
            if self.return_residual:  # mixer out is actually a pair here
                mixer_out, hidden_state = mixer_out

            hidden_state = self.norm1(mixer_out + hidden_state)  # skipping drop path

            if not isinstance(self.mlp, Identity):
                mlp_out = self.mlp(hidden_state)
                if self.return_residual:  # mlp out is actually a pair here
                    mlp_out, hidden_state = mlp_out

                hidden_state = self.norm2(mlp_out + hidden_state)  # skipping drop path

            return new_mixer_state, hidden_state

        

def create_mixer_cls(layer=None, d_model=None, n_layer=None, l_max=None, layer_kwargs=None,
                     attn_layer_idx=None, attn_cfg=None, layer_idx=None):
    if (attn_layer_idx is not None) and (layer_idx in attn_layer_idx):
        raise NotImplementedError("MHA not implemented")
        # causal = True if attn_cfg is None else attn_cfg.pop('causal', True)
        #
        # mha_cls = MHA
        #
        # mixer_cls = partial(mha_cls, causal=causal, layer_idx=layer_idx,
        #                     **(attn_cfg if attn_cfg is not None else {}),**factory_kwargs)
    elif layer is not None:
        if layer.lower() == "hyena":
            # mixer_cls = instantiate(registry.layer, layer, partial=True, layer_idx=layer_idx, **factory_kwargs)
            mixer_cls = HyenaOperator(d_model, n_layer, l_max, **layer_kwargs)

        elif layer.lower() == "s5_operator":
            mixer_cls = S5Operator(d_model, n_layer, l_max, **layer_kwargs)
        
        elif layer.lower() == "identity":
            mixer_cls = S5Operator(d_model, n_layer, l_max, filter_cls='identity', **layer_kwargs)

        else:
            raise ValueError(
                f"Expected layer to be one of 'hyena' or 's5_operator', but got {layer}."
            )
    else:
        raise ValueError(
            "Expected either attn_layer_idx or layer to not be None, but got "
            +f"attn_layer_idx={attn_layer_idx}, layer_idx={layer_idx}, and layer={layer}."
        )

    return mixer_cls


def create_mlp_cls(d_model, d_inner=None):
    inner_dim = d_inner if (d_inner is not None) else 4 * d_model

    mlp_cls = partial(Mlp,
                      hidden_features=inner_dim,
                      activation=partial(nn.gelu, approximate=True)
                      )

    return mlp_cls


def create_block(d_model, n_layer, l_max=None, layer_kwargs=None, d_inner=None,
                 layer=None, attn_layer_idx=None,
                 attn_cfg=None, layer_norm_epsilon=1e-5,
                 resid_dropout1=0.0, resid_dropout2=0.0,
                 layer_idx=None):

    mixer_cls = create_mixer_cls(layer=layer,
                                 d_model=d_model,
                                 n_layer=n_layer,
                                 l_max=l_max,
                                 layer_kwargs=layer_kwargs,
                                 attn_layer_idx=attn_layer_idx,
                                 attn_cfg=attn_cfg,
                                 layer_idx=layer_idx
                                 )
    mlp_cls = create_mlp_cls(d_model, d_inner=d_inner)
    norm_cls = partial(nn.LayerNorm, epsilon=layer_norm_epsilon)

    block = Block(d_model,
                  n_layer,
                  mixer_cls,
                  mlp_cls,
                  norm_cls=norm_cls,
                  prenorm=True,
                  resid_dropout1=resid_dropout1,
                  resid_dropout2=resid_dropout2,
                  )
    block.layer_idx = layer_idx

    return block


class LMBackbone(nn.Module):
    """Language model backbone module.

    Parameters
    ----------
    d_model: int
        Word embedding dimension
    n_layer: int
        Number of SSM blocks
    d_inner: int
        Model hidden state dimension. Harcoded internally to be 4*d_model.
    layer: str, optional. default=None
        Filter class for SSM blocks, one of {'hyena', 'S5_operator'}
        If None, uses multi-headed attention; other parameters required.
    l_max: int, optional. default: None
        Maximum input sequence length
    layer_kwargs: dict, optional. default: None
        Additional layer keyword args, passed to HyenaOperator or S5Operator.
    process_group: int, optional. default: None
        TODO not used
    attn_layer_idx: int, optional. default: None
        Attention layer index, used for multi-headed attention
    attn_cfg: dict, optional. default: None
        Attention layer configurations, used for multi-headed attention
    resid_dropout: float, optional. default: 0.0
        Residual dropout rate.
    embed_dropout: float, optional. default: 0.1
        Residual dropout rate for input embedding (i.e. first dropout of first layer)
    layer_norm_epsilon: float, optional. default: 1e-5
        Layer norm epsilon, added to variance to avoid DBZ.
    initializer_cfg: dict, optional. default: None
        TODO not used

    Arguments
    ---------
    hidden_states: ndarray, shape (d_model,)
    training: bool
        If True, dropout at specified rate. Else, turn off dropout.

    Returns
    -------
    hidden_states, ndarray, shape (d_model,)

    """

    d_model: int
    n_layer: int
    d_inner: int
    layer: str = None
    l_max: int = None
    layer_kwargs: dict = None
    process_group: int = None
    attn_layer_idx: int = None
    attn_cfg: dict = None
    resid_dropout: float = 0.0
    embed_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_cfg: dict = None

    def setup(self):

        self.layers = [
            create_block(
                self.d_model,
                self.n_layer,
                l_max=self.l_max,
                layer_kwargs=self.layer_kwargs,
                d_inner=self.d_inner,
                layer=self.layer,
                attn_layer_idx=self.attn_layer_idx,
                attn_cfg=self.attn_cfg,
                layer_norm_epsilon=self.layer_norm_epsilon,
                resid_dropout1=self.embed_dropout if (i == 0) else self.resid_dropout,
                resid_dropout2=self.resid_dropout,
                layer_idx=i
            ) for i in range(self.n_layer)
        ]

        self.drop_f = partial(nn.Dropout, self.resid_dropout)
        self.ln_f = nn.LayerNorm(epsilon=self.layer_norm_epsilon)

    @nn.compact
    def __call__(self, hidden_states, training=True):
        
        residual=None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, training, residual)

        dropped = self.drop_f(deterministic=not training)(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.ln_f(residual)

        return hidden_states, residual

    def step(self, layer_states, hidden_state):
        """Apply LMBackbone to a single input.
        
        Args
            layer_states: jax.Array, (n_layers, bsz,...)
                State of each layer (block).
            hidden_state: jax.array, (bsz, d_inner). Equivalent to input_embeddings
        
        Returns:
            new_layer_states: jax.Array, (n_layers, bsz,...)
            hidden_state: jax.array, (bsz, d_inner)
        """
        residual = None

        new_layer_states = []
        for layer, layer_state in zip(self.layers, layer_states):
            new_layer_state, hidden_state, residual = layer.step(layer_state, hidden_state, residual)
            new_layer_states.append(new_layer_state)
        new_layer_states = np.array(new_layer_states)

        dropped = hidden_state
        residual = (dropped + residual) if residual is not None else dropped
        hidden_state = self.ln_f(residual)
        return new_layer_states, hidden_state


class SimpleLMHeadModel(nn.Module):
    """Simple language model head model.

    Parameters
    ----------
    d_model: int
        Word embedding dimension
    n_layer: int
        Number of SSM blocks
    d_inner: int
        Model hidden state dimension.
    vocab_size: int
        Input vocabulary size
    layer: str, optional. default=None
        Filter class for SSM blocks, one of {'hyena', 'S5_operator'}
        If None, uses multi-headed attention; other parameters required.
    l_max: int, optional. default: None
        Maximum input sequence length
    layer_kwargs: dict, optional. default: None
        Additional layer keyword args, passed to HyenaOperator or S5Operator.
    attn_layer_idx: int, optional. default: None
        Attention layer index, used for multi-headed attention
    attn_cfg: dict, optional. default: None
        Attention layer configurations, used for multi-headed attention
    max_position_embeddings: int, optional. default: 0
        Positional embedding size. If <=0, positional embedding is not used.
    resid_dropout: float, optional. default: 0.0
        Residual dropout rate.
    embed_dropout: float, optional. default: 0.1
        Residual dropout rate for input embedding (i.e. first dropout of first layer)
    layer_norm_epsilon: float, optional. default: 1e-5
        Layer norm epsilon, added to variance to avoid DBZ.
    initializer_cfg: dict, optional. default: None
        TODO not used
    pad_vocab_size_multiple: int, optional. default: 1
        Value that vocab_size must be a multiple of. If not, pad vocab_size up to this value.
    aux_rng_streams: Sequence[str], default=('dropout',)
        Names of auxilary variable collections to seed RNG streams for.

    Arguments
    ---------
    hidden_states: ndarray, shape (d_model,)
    training: bool
        If True, dropout at specified rate. Else, turn off dropout.

    Returns
    -------
    CausalLMOutput: namedtuple, with fields
        logits: ndarray, shape (vocab_size,). Pseudo-logits of next word.
    None
    
    """

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
    aux_rng_streams = ('dropout',)

    def setup(self):
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            mod = (self.vocab_size % self.pad_vocab_size_multiple)
            self.vocab_size += (self.pad_vocab_size_multiple - mod)

        self.embeddings = GPT2Embeddings(
            self.d_model, self.vocab_size, self.max_position_embeddings
        )

        self.backbone = LMBackbone(
            d_model=self.d_model,
            n_layer=self.n_layer,
            d_inner=self.d_inner,
            layer=self.layer,
            l_max=self.l_max,
            layer_kwargs=self.layer_kwargs,
            attn_layer_idx=self.attn_layer_idx,
            attn_cfg=self.attn_cfg,
            resid_dropout=self.resid_dropout,
            embed_dropout=self.embed_dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_cfg=self.initializer_cfg
        )

    def __call__(self, input_ids, training=True, position_ids=None, state=None):

        input_embeddings = self.embeddings(input_ids, position_ids=position_ids)
        hidden_states = self.backbone(input_embeddings, training)
        logits = self.embeddings.attend(hidden_states)

        CausalLMOutput = namedtuple('CausalLMOutput', ['logits'])
        return CausalLMOutput(logits=logits), None
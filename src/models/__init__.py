from .simple_lm import SimpleLMHeadModel


def load_ckpt(ckpt_path, replicate=True, return_config=False,
              default_if_none=dict(), need_encode=None, **kwargs):
    import os.path as osp
    import pickle
    from flax import jax_utils
    from flax.training import checkpoints
    from ..train_utils import TrainState

    config = pickle.load(open(osp.join(ckpt_path, 'args'), 'rb'))
    for k, v in kwargs.items():
        setattr(config, k, v)
    for k, v in default_if_none.items():
        if not hasattr(config, k):
            print('did not find', k, 'setting default to', v)
            setattr(config, k, v)

    model = get_model(config)
    state = checkpoints.restore_checkpoint(osp.join(ckpt_path, 'checkpoints'), None)
    state = TrainState(
        step=state['step'],
        params=state['params'],
        opt_state=state['opt_state'],
        model_state=state['model_state'],
        apply_fn=model.apply,
        tx=None
    )

    assert state is not None, f'No checkpoint found in {ckpt_path}'

    if replicate:
        state = jax_utils.replicate(state)

    if return_config:
        return model, state, config
    else:
        return model, state


def get_model(config):

    if config.model == 'hyena_simplelm':
        model = SimpleLMHeadModel(config.d_model,
                                  config.n_layer,
                                  config.d_inner,
                                  config.vocab_size,
                                  layer=config.layer,
                                  l_max=config.l_max,
                                  layer_kwargs=config.layer_kwargs,
                                  **config.seq_model_kwargs)
    else:
        raise ValueError(f'Invalid model: {config.model}')

    return model

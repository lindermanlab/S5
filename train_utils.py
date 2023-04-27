from typing import Any
from collections import OrderedDict
import random
import numpy as np
import jax
from flax.training import train_state
from flax.core.frozen_dict import freeze
import optax
from functools import partial


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def map_nested_fn_with_keyword(keyword_1, keyword_2):
    '''labels all the leaves that are descendants of keyword_1 with keyword 1,
    else label the leaf with keyword_2'''

    def map_fn(nested_dict):
        output_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                if k == keyword_1:
                    output_dict[k] = map_fn_2(v)
                else:
                    output_dict[k] = map_fn(v)
            else:
                if k == keyword_1:
                    output_dict[k] = keyword_1
                else:
                    output_dict[k] = keyword_2
        return output_dict

    def map_fn_2(nested_dict):
        output_dict = {}
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                output_dict[k] = map_fn_2(v)
            else:
                output_dict[k] = keyword_1
        return output_dict

    return map_fn


@partial(jax.jit, static_argnums=(1))
def reshape_batch_per_device(x, num_devices):
    batch_size_per_device, ragged = divmod(x.shape[0], num_devices)
    if ragged:
        msg = "batch size must be divisible by device count, got {} and {}."
        raise ValueError(msg.format(x.shape[0], num_devices))
    return x.reshape((num_devices, batch_size_per_device, ) + (x.shape[1:]))


class TrainState(train_state.TrainState):
    model_state: Any


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def get_first_device(x):
    x = jax.tree_util.tree_map(lambda a: a[0], x)
    return jax.device_get(x)


def print_model_size(params, name=''):
    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    total_params_size = sum(jax.tree_leaves(param_sizes))
    print('model parameter count:', total_params_size)


def get_learning_rate_fn(config, lr):
    if config.lr_schedule == 'cosine':
        learning_rate_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.,
            peak_value=lr,
            warmup_steps=config.warmup_steps,
            decay_steps=config.total_steps - config.warmup_steps
        )
    elif config.lr_schedule == 'constant':
        learning_rate_fn = optax.join_schedules([
            optax.linear_schedule(
                init_value=0.,
                end_value=lr,
                transition_steps=config.warmup_steps
            ),
            optax.constant_schedule(lr)
        ], [config.warmup_steps])
    else:
        raise ValueError(f'Unknown schedule: {config.lr_schedule}')

    return learning_rate_fn


def get_optimizer(config, params):
    if config.layer == "hyena":
        optimizers = {}

        optimizers["implicit_filter"] = optax.adamw(learning_rate=get_learning_rate_fn(config, config.implicit_filter_lr),
                                                    b1=0.9, b2=0.999,
                                                    weight_decay=config.implicit_filter_weight_decay)

        learning_rate_fn = get_learning_rate_fn(config, config.lr)
        optimizers["__default__"] = optax.adamw(learning_rate=learning_rate_fn, b1=0.9, b2=0.999,
                                                weight_decay=config.weight_decay)

        name_map = map_nested_fn_with_keyword("implicit_filter", "__default__")(params)
        tx = optax.multi_transform(optimizers, name_map)
        return tx, learning_rate_fn

    elif config.layer == "S5_operator":

        ssm_lrs = ["B", "Lambda_re", "Lambda_im"]
        ssm_fn = map_nested_fn(
            lambda k, _: "ssm"
            if k in ssm_lrs
            else "regular"
        )

        learning_rate_fn = get_learning_rate_fn(config, config.lr)
        tx = optax.multi_transform(
            {
                "ssm": optax.adamw(learning_rate=get_learning_rate_fn(config, config.implicit_filter_lr),
                                                    b1=0.9, b2=0.999,
                                                    weight_decay=config.implicit_filter_weight_decay),
                "regular": optax.adamw(learning_rate=learning_rate_fn, b1=0.9, b2=0.999,
                                                weight_decay=config.weight_decay),
            },
            ssm_fn,
        )

        return tx, learning_rate_fn


def init_model_state(rng_key, model, input, config):
    variables = model.init({k: rng_key for k in ['params', *config.rng_keys]}, input, training=True
    ).unfreeze()
    params = variables.pop('params')
    model_state = variables
    print_model_size(params)

    tx, learning_rate_fn = get_optimizer(config, params)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        model_state=model_state
    ), learning_rate_fn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f')
                                   for mn in meter_names})
        self.prefix = prefix

    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'
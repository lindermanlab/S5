import os
import os.path as osp
import time
import argparse
import yaml
import pickle
import wandb
import glob

import jax
from jax import random
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils

import optax
from functools import partial

from dataloading import create_wikitext_dataset, create_icl_datasets
from train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all, reshape_batch_per_device
from src.models import get_model


def main():
    global model
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    files = glob.glob(osp.join(config.output_dir, 'checkpoints', '*'))
    if len(files) > 0:
        print('Found previous checkpoints', files)
        config.ckpt = config.output_dir
    else:
        config.ckpt = None

    if is_master_process:
        root_dir = os.environ['DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)

        wandb.init(project=config.project, entity=config.entity, config=config,
                   dir=root_dir, id=config.run_id, resume='allow')
        wandb.run.name = config.run_id
        wandb.run.save()

    if config.dataset in ["wikitext103"]:
        train_loader, val_loader, test_loader = create_wikitext_dataset(config)
    elif config.dataset in ["icl_synthetics"]:
        train_loader, val_loader, test_loader = create_icl_datasets(config)
    else:
        raise NotImplementedError("Dataset not implemented")
    log_metrics = ['loss', 'accuracy']

    batch = next(iter(train_loader))
    inputs = jnp.array(batch[0].numpy())
    targets = jnp.array(batch[1].numpy())

    # Reshape to (num_devices, device_batch_size, seq_len, dim)
    num_devices = jax.local_device_count()
    inputs = reshape_batch_per_device(inputs, num_devices)
    targets = reshape_batch_per_device(targets, num_devices)
    batch = (inputs, targets)  # Just want to use 1 device batch for init

    batch = get_first_device(batch)
    model = get_model(config)
    state, schedule_fn = init_model_state(init_rng, model, batch[0], config)
    if config.ckpt is not None:
        state = checkpoints.restore_checkpoint(osp.join(config.ckpt, 'checkpoints'), state)
        print('Restored from checkpoint')

    iteration = int(state.step)
    state = jax_utils.replicate(state)

    ckpt_dir = osp.join(config.output_dir, 'checkpoints')

    rngs = random.split(rng, jax.local_device_count())
    while iteration <= config.total_steps:
        iteration, state, rngs = train(iteration, log_metrics, state, train_loader,
                                       schedule_fn, rngs, ckpt_dir)

        validate(iteration, state, val_loader, val=True)

        validate(iteration, state, test_loader)


def train_step(batch, state, rng, vocab_size):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}

    inputs = batch[0]
    targets = batch[1]

    def loss_fn(params):
        variables = {'params': params, **state.model_state}
        out = state.apply_fn(
            variables,
            inputs,
            training=True,
            rngs=rngs
        )
        out_tuple, _ = out
        logits = out_tuple.logits
        labels = jax.nn.one_hot(targets, num_classes=vocab_size)

        loss = optax.softmax_cross_entropy(logits, labels)
        loss = loss.mean()
        preds = jnp.argmax(logits, axis=-1)
        accuracy = (preds == targets).mean()
        out_dict = {'loss': loss,
                    'accuracy': accuracy}

        return loss, out_dict

    return_dict, grads = jax.value_and_grad(loss_fn,
                                            has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, return_dict[1], new_rng


def train(iteration, log_metrics, state, train_loader, schedule_fn, rngs, ckpt_dir):
    progress = ProgressMeter(config.total_steps,
                             ['time', 'data'] + log_metrics)

    num_devices = jax.local_device_count()
    p_train_step = jax.pmap(partial(train_step, vocab_size=config.vocab_size), axis_name='batch')

    end = time.time()
    for batch in train_loader:
        inputs = jnp.array(batch[0].numpy())
        targets = jnp.array(batch[1].numpy())

        # Reshape to (num_devices, device_batch_size, seq_len, dim)
        inputs = reshape_batch_per_device(inputs, num_devices)
        targets = reshape_batch_per_device(targets, num_devices)
        batch = (inputs, targets)

        batch_size = batch[0].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in log_metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            wandb.log({'train/lr': schedule_fn(iteration)}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                          for metric, val in metrics.items()}
                       }, step=iteration)

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.save_interval == 0:
            if is_master_process:
                state_ = jax_utils.unreplicate(state)
                save_path = checkpoints.save_checkpoint(ckpt_dir, state_, state_.step, keep=1)
                print('Saved checkpoint to', save_path)
                del state_  # Needed to prevent a memory leak bug

        progress.update(time=time.time() - end)
        end = time.time()

        iteration += 1

    return iteration, state, rngs


def eval_step(batch, state, vocab_size):
    inputs = batch[0]
    targets = batch[1]

    variables = {'params': state.params, **state.model_state}
    out = state.apply_fn(
        variables,
        inputs,
        training=False,
    )
    out_tuple, _ = out
    logits = out_tuple.logits
    labels = jax.nn.one_hot(targets, num_classes=vocab_size)

    loss = optax.softmax_cross_entropy(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == targets)
    return loss, accuracy


def eval_step_synthetic(batch, state, vocab_size):
    """Different eval loss functions for
       synthetic associative_recall task"""
    inputs = batch[0]
    targets = batch[1]

    variables = {'params': state.params, **state.model_state}
    out = state.apply_fn(
        variables,
        inputs,
        training=False,
    )
    out_tuple, _ = out
    logits = out_tuple.logits[:, -1]
    labels = jax.nn.one_hot(targets[:, -1], num_classes=vocab_size)

    loss = optax.softmax_cross_entropy(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = (preds == targets[:, -1])

    return loss, accuracy


def validate(iteration, state, test_loader, val=False):
    losses = jnp.array([])
    accs = jnp.array([])

    # Todo: may need to change for multinode
    num_devices = jax.local_device_count()

    if config.dataset in ["wikitext103"]:
        p_eval_step = jax.pmap(partial(eval_step, vocab_size=config.vocab_size), axis_name='batch')
    elif config.dataset in ["icl_synthetics"]:
        p_eval_step = jax.pmap(partial(eval_step_synthetic, vocab_size=config.vocab_size), axis_name='batch')
    else:
        raise NotImplementedError("Dataset not implemented")

    for batch in test_loader:
        inputs = jnp.array(batch[0].numpy())
        targets = jnp.array(batch[1].numpy())

        # Reshape to (num_devices, device_batch_size, seq_len, dim)
        inputs = reshape_batch_per_device(inputs, num_devices)
        targets = reshape_batch_per_device(targets, num_devices)
        batch = (inputs, targets)

        return_loss, return_acc = p_eval_step(batch=batch, state=state)
        losses = jnp.append(losses, return_loss)
        accs = jnp.append(accs, return_acc)

    avg_loss = jnp.mean(losses)
    avg_perplexity = jnp.exp(avg_loss)
    avg_accuracy = jnp.mean(accs)
    if is_master_process:
        if val:
            prefix = "val"
        else:
            prefix = "test"

        print(prefix+'/loss:', avg_loss)
        print(prefix + '/perplexity:', avg_perplexity)
        print(prefix + '/accuracy:', avg_accuracy)

        wandb.log({prefix+'/loss': avg_loss}, step=iteration)
        wandb.log({prefix+'/perplexity': avg_perplexity}, step=iteration)
        wandb.log({prefix + '/accuracy': avg_accuracy}, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    args.run_id = args.output_dir

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')
    print(f'JAX local devices: {jax.local_device_count()}')

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            os.environ['DATA_DIR'] = 'logs'
            print('DATA_DIR environment variable not set, default to logs/')
        root_folder = os.environ['DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['save_interval'] = 2
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    is_master_process = jax.process_index() == 0

    main()

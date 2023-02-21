from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
import wandb
from timeit import default_timer as dt

from .train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel, BatchGaussianRegressionModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    else:
        wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Define some flags for data.
    padded = False
    retrieval = False
    speech = False
    cru = False
    variable_observation_interval = False
    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")

    # For speech dataset
    if args.dataset in ["speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")

    # For CRU/Pendulum data.
    if args.dataset == "cru-image-pendulum-regression":
        cru = True

        # If weare using the integration timestep, then set the flag indicating so.
        if args.use_integration_timestep:
            variable_observation_interval = True

        # If we are appending the integration timestep, then the model isn't
        # actually a variable integration timestep model (the timestep is
        # part of the "observation") and so disable this.
        if args.append_integration_timestep:
            variable_observation_interval = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
      create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    print("Lambda.shape={}".format(Lambda.shape))
    print("V.shape={}".format(V.shape))
    print("Vinv.shape={}".format(Vinv.shape))

    ssm_init_fn = init_S5SSM(H=args.d_model,
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args.C_init,
                             discretization=args.discretization,
                             dt_min=args.dt_min,
                             dt_max=args.dt_max,
                             variable_observation_interval=variable_observation_interval,
                             conj_sym=args.conj_sym,
                             clip_eigs=args.clip_eigs,
                             bidirectional=args.bidirectional)

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    elif cru:
        print("using crn regression model")
        from s5.cru.util import CRN_CNN
        model_cls = partial(
            BatchGaussianRegressionModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
            append_integration_timestep=args.append_integration_timestep,
            use_integration_timestep=args.use_integration_timestep,
            encoder_fn=lambda d: CRN_CNN(d, input_shape=24, append_timestep=args.append_integration_timestep)
        )

    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            bn_momentum=args.bn_momentum,
        )

    # initialize training state
    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               retrieval,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(train_size/args.bsz)
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        st = dt()
        state, train_loss, step, sum_train_step_time = train_epoch(state,
                                                                   skey,
                                                                   model_cls,
                                                                   trainloader,
                                                                   seq_len,
                                                                   in_dim,
                                                                   args.batchnorm,
                                                                   lr_params,
                                                                   cru=cru)
        _ = train_loss.block_until_ready()
        outer_train_step_time = dt() - st

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            st = dt()
            val_loss, val_acc, sum_eval_step_time = validate(state,
                                                             model_cls,
                                                             valloader,
                                                             seq_len,
                                                             in_dim,
                                                             args.batchnorm,
                                                             cru=cru)
            blocked = val_loss.block_until_ready()
            outer_eval_time = dt() - st
            print("Outer epoch evaluation time: ", outer_eval_time, blocked)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            st = dt()
            test_loss, test_acc, sum_eval_step_time = validate(state,
                                                               model_cls,
                                                               testloader,
                                                               seq_len,
                                                               in_dim,
                                                               args.batchnorm,
                                                               cru=cru)
            blocked = test_loss.block_until_ready()
            outer_eval_time = dt() - st
            print("Outer epoch evaluation time: ", outer_eval_time, blocked)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            st = dt()
            val_loss, val_acc, sum_eval_step_time = validate(state,
                                                             model_cls,
                                                             testloader,
                                                             seq_len,
                                                             in_dim,
                                                             args.batchnorm,
                                                             cru=cru)
            blocked = val_loss.block_until_ready()
            outer_eval_time = dt() - st
            print("Outer epoch evaluation time: ", outer_eval_time, blocked)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # Do some validation on improvement.
            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc, _ = validate(state,
                                                  model_cls,
                                                  aux_dataloaders['valloader2'],
                                                  int(seq_len // 2),
                                                  in_dim,
                                                  args.batchnorm,
                                                  step_rescale=2.0)

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc, _ = validate(state, model_cls, aux_dataloaders['testloader2'], int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if valloader is not None:
            if speech:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val Loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Val2 loss": val2_loss,
                        "Val2 Accuracy": val2_acc,
                        "Test2 Loss": test2_loss,
                        "Test2 Accuracy": test2_acc,
                        "Best Test Accuracy": best_test_acc,
                        "Best Test Loss": best_test_loss,
                        "Opt acc": opt_acc,
                        "lr/count": count,
                        "lr/Learning rate count": lr_count,
                        "lr/lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "lr/ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
                        "timing/sum_eval_step_time": sum_eval_step_time,
                        "timing/outer_eval_time": outer_eval_time,
                        "timing/sum_train_step_time": sum_train_step_time,
                        "timing/outer_train_step_time": outer_train_step_time,
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val Loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Best Test Accuracy": best_test_acc,
                        "Best Test Loss": best_test_loss,
                        "Opt acc": opt_acc,
                        "lr/count": count,
                        "lr/Learning rate count": lr_count,
                        "lr/lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "lr/ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
                        "timing/sum_eval_step_time": sum_eval_step_time,
                        "timing/outer_eval_time": outer_eval_time,
                        "timing/sum_train_step_time": sum_train_step_time,
                        "timing/outer_train_step_time": outer_train_step_time,
                    }
                )

        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val Loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Best Test Accuracy": best_test_acc,
                    "Best Test Loss": best_test_loss,
                    "Opt acc": opt_acc,
                    "lr/count": count,
                    "lr/Learning rate count": lr_count,
                    "lr/lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                    "lr/ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'],
                    "timing/sum_eval_step_time": sum_eval_step_time,
                    "timing/outer_eval_time": outer_eval_time,
                    "timing/sum_train_step_time": sum_train_step_time,
                    "timing/outer_train_step_time": outer_train_step_time,
                }
            )
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            break

WANDB_ENTITY=<INSERT_WANDB_USER_NAME> # Insert WandB username or set USE_WANDB=0

EPOCHS=100
WANDB_PROJ=pendulum-1
SEED=1

echo S5
python3.9 run_train.py --method 'S5' --C_init=trunc_standard_normal --USE_WANDB=True --activation_fn=half_glu2 --batchnorm=False --bidirectional=False --blocks=8 --bn_momentum=0.95 --bsz=32 --clip_eigs=False --conj_sym=True --cosine_anneal=True --d_model=30 --dataset=cru-image-pendulum-regression --dir_name=./cache_dir --discretization=zoh --dt_global=False --dt_max=0.1 --dt_min=0.01 --early_stop_patience=100 --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=4 --lr_min=0 --lr_patience=1000 --mode=pool --n_layers=4 --opt_config=standard --p_dropout=0 --prenorm=False --reduce_factor=1 --ssm_lr=0.003 --ssm_size=16 --wandb_entity=$WANDB_ENTITY --wandb_project=$WANDB_PROJ --warmup_end=1 --weight_decay=0  

echo S5-ignore
python3.9 run_train.py --method 'S5-ignore' --C_init=trunc_standard_normal --USE_WANDB=True --activation_fn=half_glu2 --batchnorm=False --bidirectional=False --blocks=8 --bn_momentum=0.95 --bsz=32 --clip_eigs=False --conj_sym=True --cosine_anneal=True --d_model=30 --dataset=cru-image-pendulum-regression --dir_name=./cache_dir --discretization=zoh --dt_global=False --dt_max=0.1 --dt_min=0.01 --early_stop_patience=100 --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=4 --lr_min=0 --lr_patience=1000 --mode=pool --n_layers=4 --opt_config=standard --p_dropout=0 --prenorm=False --reduce_factor=1 --ssm_lr=0.003 --ssm_size=16 --wandb_entity=$WANDB_ENTITY --wandb_project=$WANDB_PROJ --warmup_end=1 --weight_decay=0 --use_integration_timestep False

echo S5-append
python3.9 run_train.py --method 'S5-append' --C_init=trunc_standard_normal --USE_WANDB=True --activation_fn=half_glu2 --batchnorm=False --bidirectional=False --blocks=8 --bn_momentum=0.95 --bsz=32 --clip_eigs=False --conj_sym=True --cosine_anneal=True --d_model=30 --dataset=cru-image-pendulum-regression --dir_name=./cache_dir --discretization=zoh --dt_global=False --dt_max=0.1 --dt_min=0.01 --early_stop_patience=100 --epochs=$EPOCHS --jax_seed=$SEED --lr_factor=4 --lr_min=0 --lr_patience=1000 --mode=pool --n_layers=4 --opt_config=standard --p_dropout=0 --prenorm=False --reduce_factor=1 --ssm_lr=0.003 --ssm_size=16 --wandb_entity=$WANDB_ENTITY --wandb_project=$WANDB_PROJ --warmup_end=1 --weight_decay=0 --use_integration_timestep True --append_integration_timestep True

echo CRU
cd ./Continuous-Recurrent-Units
python3.9 run_experiment.py --method 'CRU' --dataset pendulum --task regression -lsd 30 --sample-rate 0.5 --epochs $EPOCHS --wandb-project $WANDB_PROJ --wandb-entity $WANDB_ENTITY --random-seed $SEED --num-workers 1 --dir_name ./../cache_dir
cd ..


python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 \
                    --batchnorm=True --bidirectional=False --blocks=1 --bsz=50 \
                    --d_model=128 --dataset=imdb-classification \
                    --dt_global=False --epochs=50 --jax_seed=8825365 --lr_factor=3 \
                    --n_layers=4 --opt_config=standard --p_dropout=0.1 --ssm_lr_base=0.001 \
                    --ssm_size_base=8 --warmup_end=0 --weight_decay=0.01 --liquid=True
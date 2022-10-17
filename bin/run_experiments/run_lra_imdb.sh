python run_train.py --C_init=lecun_normal --activation_fn=half_glu2 \
                    --batchnorm=True --bidirectional=True --blocks=12 --bsz=50 \
                    --d_model=256 --dataset=imdb-classification \
                    --dt_global=True --epochs=35 --jax_seed=8825365 --lr_factor=4 \
                    --n_layers=6 --opt_config=standard --p_dropout=0.1 --ssm_lr_base=0.001 \
                    --ssm_size_base=192 --warmup_end=0 --weight_decay=0.07

WANDB_PROJ='pendulum-timing-1'

for SEED in `seq 0 19`
do
    sbatch --job-name=pendulum-$SEED --export=ALL,WANDB_PROJ=$WANDB_PROJ,SEED=$SEED submit_pendulum_timing.sbatch
done



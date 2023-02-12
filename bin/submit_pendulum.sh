
WANDB_PROJ='pendulum-public-2'

for SEED in `seq 0 19`
do
    sbatch --job-name=pendulum-$SEED --export=ALL,WANDB_PROJ=$WANDB_PROJ,SEED=$SEED submit_pendulum.sbatch
done



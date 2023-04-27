# S5: Simplified State Space Layers for Sequence Modeling

This repository provides the implementation for the
paper:

**Simplified State Space Layers for Sequence Modeling**  
Jimmy T.H. Smith\*, Andrew Warrington\*, Scott Linderman  
International Conference on Learning Representations, 2023.  
Notable-top-5% (Oral).  
[arXiv](https://arxiv.org/abs/2208.04933)  
[OpenReview](https://openreview.net/forum?id=Ai8Hw3AXqks)

![](./docs/figures/pngs/s5-matrix-blocks.png)
<p style="text-align: center;">
Figure 1:  S5 uses a single multi-input, multi-output linear state-space model, coupled with non-linearities, to define a non-linear sequence-to-sequence transformation. Parallel scans are used for efficient offline processing. 
</p>


The S5 layer builds on the prior S4 work ([paper](https://arxiv.org/abs/2111.00396)). While it has departed considerably, this repository originally started off with much of the JAX implementation of S4 from the
Annotated S4 blog by Rush and Karamcheti (available [here](https://github.com/srush/annotated-s4)).


## Requirements & Installation
To run the code on your own machine, run either `pip install -r requirements_cpu.txt` or `pip install -r requirements_gpu.txt`.  The GPU installation of JAX can be tricky, and so we include requirements that should work for most people, although further instructions are available [here](https://github.com/google/jax#installation).

Run from within the root directory `pip install -e .` to install the package. 


## Data Download
Downloading the raw data is done differently for each dataset.  The following datasets require no action:
- Text (IMDb)
- Image (Cifar black & white)
- sMNIST
- psMNIST
- Cifar (Color)

The remaining datasets need to be manually downloaded.  To download _everything_, run `./bin/download_all.sh`.  This will download quite a lot of data and will take some time.  

Below is a summary of the steps for each dataset:
- ListOps: run `./bin/download_lra.sh` to download the full LRA dataset.  
- Retrieval (AAN): run `./bin/download_aan.sh`
- Pathfinder: run `./bin/download_lra.sh` to download the full LRA dataset.
- Path-X: run `./bin/download_lra.sh` to download the full LRA dataset.
- Speech commands 35: run `./bin/download_sc35.sh` to download the speech commands data.

*With the exception of SC35.*  When the dataset is used for the first time, a cache is created in `./cache_dir`.  Converting the data (e.g. tokenizing) can be quite slow, and so this cache contains the processed dataset.  The cache can be moved and specified with the `--dir_name` argument (i.e. the default is `--dir_name=./cache_dir`) to avoid applying this preprocessing every time the code is run somewhere new.

SC35 is slightly different.  SC35 doesn't use `--dir_name`, and instead requires that the following path exists: `./raw_datasets/speech_commands/0.0.2/SpeechCommands` (i.e. the directory `./raw_datasets/speech_commands/0.0.2/SpeechCommands/zero` must exist).  The cache is then stored in `./raw_datasets/speech_commands/0.0.2/SpeechCommands/processed_data`.  This directory can then be copied (preserving the directory path) to move the preprocessed dataset to a new location.


## Repository Structure
Directories and files that ship with GitHub repo:
```
s5/                    Source code for models, datasets, etc.
    dataloading.py          Dataloading functions.
    layers.py               Defines the S5 layer which wraps the S5 SSM with nonlinearity, norms, dropout, etc.
    seq_model.py            Defines deep sequence models that consist of stacks of S5 layers.
    ssm.py                  S5 SSM implementation.
    ssm_init.py             Helper functions for initializing the S5 SSM .
    train.py                Training loop code.
    train_helpers.py        Functions for optimization, training and evaluation steps.
    dataloaders/            Code mainly derived from S4 processing each dataset.
    utils/                  Range of utility functions.
bin/                    Shell scripts for downloading data and running example experiments.
requirements_cpu.txt    Requirements for running in CPU mode (not advised).
requirements_gpu.txt    Requirements for running in GPU mode (installation can be highly system-dependent).
run_train.py            Training loop entrypoint.
```

Directories that may be created on-the-fly:
```
raw_datasets/       Raw data as downloaded.
cache_dir/          Precompiled caches of data.  Can be copied to new locations to avoid preprocessing.
wandb/              Local WandB log files.
```

## Experiments

The configurations to run the LRA and 35-way Speech Commands experiments from the paper are located in  `bin/run_experiments`. For example,
to run the LRA text (character level IMDB) experiment, run `./bin/run_experiments/run_lra_imdb.sh`. 
To log with W&B, adjust the default `USE_WANDB, wandb_entity, wandb_project` arguments. 
Note: the pendulum
regression dataloading and experiments are on the separate [pendulum](https://github.com/lindermanlab/S5/tree/pendulum) branch.

## Language Modeling
Check out the [development](https://github.com/lindermanlab/S5/tree/development) branch, where we have started to add some preliminary language modeling tasks such 
as WikiText-103 and the synthetic in-context learning tasks from the [H3](https://arxiv.org/abs/2212.14052) and [Hyena](https://arxiv.org/abs/2302.10866) papers.
We have found simply replacing Hyena's implicitly parameterized convolutions with S5 SSMs provides strong performance on these tasks.
We hope to add larger scale experiments and plan to merge some of the new training infrastructure into the main branch soon!


## Citation
Please use the following when citing our work:
```
@inproceedings{
smith2023simplified,
title={Simplified State Space Layers for Sequence Modeling},
author={Jimmy T.H. Smith and Andrew Warrington and Scott Linderman},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Ai8Hw3AXqks}
}
```

Please reach out if you have any questions.

-- The S5 authors.

# S5: Simplified State Space Layers for Sequence Modeling

##Overview
This repository provides the official implementation and experiments for the 
paper: Simplified State Space Layers for Sequence Modeling.  The preprint is available [here](https://arxiv.org/abs/2208.04933). 
The core contribution is the S5 layer which is meant to simplify the prior
S4 approach [paper](https://arxiv.org/abs/2111.00396) while retaining its performance and efficiency.

While it has departed a fair amount, this repository originally started off with much of the JAX implementation of S4 from the
Annotated S4 blog post by Sasha Rush (available [here](https://github.com/srush/annotated-s4)). 

##Work in progress 09/06/2022:
This repository is in active development. A more complete release will be made available by the end of September 2022.


## Experiments
The Long Range Arena and 
Speech Commands (10-way) experiments in the paper were performed using the dataloaders from the [Official S4 repository](https://github.com/HazyResearch/state-spaces). 
We are currently in the process of adding dataloaders better suited for our JAX implementation.

We currently provide the ability to run the LRA-Text (character level IMDb classification) experiment easily in a Google Colab notebook (<a href="https://githubtocolab.com/lindermanlab/S5/blob/main/Examples/S5_IMDB_Classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>). 
The rest of the experiments will be added soon!


## Requirements
If using the Google Colab environment, all requirements are already installed.

To run the code on your own machine, you will need to first install JAX following the instructions at: https://github.com/google/jax#installation, 
before installing the requirements.txt file.


## Repository Structure
```
data/            default location of data files
Examples/        includes Colab Notebook examples of experiments
src/             source code for models, datasets, etc.
    dataloading.py   dataloading functions
    layers.py        Defines the S5 layer which wraps the S5 SSM with nonlinearity, norms, dropout, etc.
    seq_model.py     Defines deep sequence models that consist of stacks of S5 layers
    ssm.py           S5 SSM implementation
    ssm_init.py      Helper functions for initializing the S5 SSM 
    train.py         training loop entrypoint
    train_helpers.py functions for optimization, training and evaluation steps
```

## Citation
Please use the following when citing our work:
```
@article{smith2022S5,
  title={Simplified State Space Layers for Sequence Modeling},
  author={Smith, Jimmy T.H. and Warrington, Andrew and Linderman, Scott},
  journal={arXiv preprint arXiv:2208.04933},
  year={2022}
}
```
Please reach out if you have any questions.  

-- The S5 authors.

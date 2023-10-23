import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union

DEFAULT_CACHE_DIR_ROOT = Path('./cache_dir/')

DataLoader = TypeVar('DataLoader')
InputType = [str, Optional[int], Optional[int]]
ReturnType = Tuple[DataLoader, DataLoader, DataLoader, Dict, int, int, int, int]

# Custom loading functions must therefore have the template.
dataset_fn = Callable[[str, Optional[int], Optional[int]], ReturnType]


# Example interface for making a loader.
def custom_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42) -> ReturnType:
	...


def make_data_loader(dset,
					 dobj,
					 seed: int,
					 batch_size: int=128,
					 shuffle: bool=True,
					 drop_last: bool=True,
					 collate_fn: callable=None):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:return:
	"""

	# Create a generator for seeding random number draws.
	if seed is not None:
		rng = torch.Generator()
		rng.manual_seed(seed)
	else:
		rng = None

	if dobj is not None:
		assert collate_fn is None
		collate_fn = dobj._collate_fn

	# Generate the dataloaders.
	return torch.utils.data.DataLoader(dataset=dset, collate_fn=collate_fn, batch_size=batch_size, shuffle=shuffle,
									   drop_last=drop_last, generator=rng)


def create_lra_imdb_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										   bsz: int = 50,
										   seed: int = 42) -> ReturnType:
	"""

	:param cache_dir:		(str):		Not currently used.
	:param bsz:				(int):		Batch size.
	:param seed:			(int)		Seed for shuffling data.
	:return:
	"""
	print("[*] Generating LRA-text (IMDB) Classification Dataset")
	from s5.dataloaders.lra import IMDB
	name = 'imdb'

	dataset_obj = IMDB('imdb', )
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trainloader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	testloader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	valloader = None

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 135  # We should probably stop this from being hard-coded.
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trainloader, valloader, testloader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_listops_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											  bsz: int = 50,
											  seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.lra import ListOps
	name = 'listops'
	dir_name = './raw_datasets/lra_release/lra_release/listops-1000'

	dataset_obj = ListOps(name, data_dir=dir_name)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = 20
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_path32_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											 bsz: int = 50,
											 seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-Pathfinder32 Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 32
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_pathx_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											bsz: int = 50,
											seed: int = 42) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-PathX Classification Dataset")
	from s5.dataloaders.lra import PathFinder
	name = 'pathfinder'
	resolution = 128
	dir_name = f'./raw_datasets/lra_release/lra_release/pathfinder{resolution}'

	dataset_obj = PathFinder(name, data_dir=dir_name, resolution=resolution)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = dataset_obj.d_input
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_image_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
											seed: int = 42,
											bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating LRA-listops Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': True,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)  # TODO - double check what the dir here does.
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_lra_aan_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										  bsz: int = 50,
										  seed: int = 42, ) -> ReturnType:
	"""
	See abstract template.
	"""
	print("[*] Generating LRA-AAN Classification Dataset")
	from s5.dataloaders.lra import AAN
	name = 'aan'

	dir_name = './raw_datasets/lra_release/lra_release/tsv_data'

	kwargs = {
		'n_workers': 1,  # Multiple workers seems to break AAN.
	}

	dataset_obj = AAN(name, data_dir=dir_name, **kwargs)
	dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.l_max
	IN_DIM = len(dataset_obj.vocab)
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_speechcommands35_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
												   bsz: int = 50,
												   seed: int = 42) -> ReturnType:
	"""
	AG inexplicably moved away from using a cache dir...  Grumble.
	The `cache_dir` will effectively be ./raw_datasets/speech_commands/0.0.2 .

	See abstract template.
	"""
	print("[*] Generating SpeechCommands35 Classification Dataset")
	from s5.dataloaders.basic import SpeechCommands
	name = 'sc'

	dir_name = f'./raw_datasets/speech_commands/0.0.2/'
	os.makedirs(dir_name, exist_ok=True)

	kwargs = {
		'all_classes': True,
		'sr': 1  # Set the subsampling rate.
	}
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = dataset_obj.dataset_train.tensors[0].shape[1]
	IN_DIM = 1
	TRAIN_SIZE = dataset_obj.dataset_train.tensors[0].shape[0]

	# Also make the half resolution dataloader.
	kwargs['sr'] = 2
	dataset_obj = SpeechCommands(name, data_dir=dir_name, **kwargs)
	dataset_obj.setup()
	val_loader_2 = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader_2 = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	aux_loaders = {
		'valloader2': val_loader_2,
		'testloader2': tst_loader_2,
	}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_cifar_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
										seed: int = 42,
										bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating CIFAR (color) Classification Dataset")
	from s5.dataloaders.basic import CIFAR10
	name = 'cifar'

	kwargs = {
		'grayscale': False,  # LRA uses a grayscale CIFAR image.
	}

	dataset_obj = CIFAR10(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 32 * 32
	IN_DIM = 3
	TRAIN_SIZE = len(dataset_obj.dataset_train)

	aux_loaders = {}

	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_mnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': False
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE


def create_pmnist_classification_dataset(cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR_ROOT,
																				seed: int = 42,
																				bsz: int=128) -> ReturnType:
	"""
	See abstract template.

	Cifar is quick to download and is automatically cached.
	"""

	print("[*] Generating permuted-MNIST Classification Dataset")
	from s5.dataloaders.basic import MNIST
	name = 'mnist'

	kwargs = {
		'permute': True
	}

	dataset_obj = MNIST(name, data_dir=cache_dir, **kwargs)
	dataset_obj.setup()

	trn_loader = make_data_loader(dataset_obj.dataset_train, dataset_obj, seed=seed, batch_size=bsz)
	val_loader = make_data_loader(dataset_obj.dataset_val, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)
	tst_loader = make_data_loader(dataset_obj.dataset_test, dataset_obj, seed=seed, batch_size=bsz, drop_last=False, shuffle=False)

	N_CLASSES = dataset_obj.d_output
	SEQ_LENGTH = 28 * 28
	IN_DIM = 1
	TRAIN_SIZE = len(dataset_obj.dataset_train)
	aux_loaders = {}
	return trn_loader, val_loader, tst_loader, aux_loaders, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE

# Code with regards to the BCI dataset (no updates from meeting 10/16)
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import timeit
import itertools
from scipy.io import loadmat
from array import array


class BCIDataset(Dataset):
  def __init__(self, path):
    # data loading
    xy = loadmat(path, squeeze_me=True)

    # Figure out padding, to create array to input
    def stack_padding(it):
      def resize(row, size):
          new = np.array(row)
          # This also restricts are array of interest.
          new.resize(128, size)
          return new

      # find longest row length
      row_length = max(it[:], key=len).__len__()
      mat = np.array( [resize(row, row_length) for row in it] )
      return mat

    # Pytorch cannot create a tensor from strings so convert to ASCHII
    self.setenceText = []
    for sentence in xy['sentenceText'][:]:
      self.setenceText.append(np.array(list(map(ord, sentence))))
      # Assign each character to an integers, use the same punctation that nptl uess.
      # Look up the dictionary that they're using to get the sentence phonemes. (At sometime)
    self.setenceText = torch.from_numpy(np.array(self.setenceText))


    self.tx1 = torch.from_numpy(stack_padding(np.array(xy['tx1'])))
    self.spikePow = torch.from_numpy(stack_padding(np.array(xy['spikePow'])))
    # padding of feature which is passed on (For each sentence there will be one length)
    # 0 is valid and 1 is padded. Pad sentences
    self.n_samples = xy['sentenceText'].shape[0]

  # Output padding as aux_data
  def __getitem__(self, index):
    return [self.tx1[index], self.spikePow[index]], self.setenceText[index]

  def __len__(self):
    return self.n_samples

# Example interface for making a loader.
def BCIData_loader(cache_dir: str,
				  bsz: int = 50,
				  seed: int = 42,
          shuffle: bool = True):

  train_str = str(cache_dir) + "train/t12.2022.05.24.mat"
  test_str = str(cache_dir) + "test/t12.2022.05.24.mat"

  trainDataset = BCIDataset(path=train_str)
  testDataset = BCIDataset(path=test_str)

  trainloader = DataLoader(dataset=trainDataset, batch_size=bsz)
  testloader = DataLoader(dataset=testDataset, batch_size=bsz, shuffle=False)
  valloader = None

  # Stack the tx1 and spikePow earlier
  neuralData_train, labels_train = trainDataset[0]
  neuralData_test, labels_test = testDataset[0]

  # Not sure what these variables exactly code for.
  N_CLASSES = len(labels_train) + len(labels_test)
  # Would be one or the other
  SEQ_LENGTH = max(len(tx1_train[0]),len(tx1_test[0]))
  IN_DIM = 256  # Guessing
  TRAIN_SIZE = len(labels_train)
  aux_loader = {}

  return trainloader, valloader, testloader, aux_loader, N_CLASSES, SEQ_LENGTH, IN_DIM, TRAIN_SIZE

Datasets = {
	# Other loaders.
	"mnist-classification": create_mnist_classification_dataset,
	"pmnist-classification": create_pmnist_classification_dataset,
	"cifar-classification": create_cifar_classification_dataset,
  "BCI-classification": BCIData_loader,

	# LRA.
	"imdb-classification": create_lra_imdb_classification_dataset,
	"listops-classification": create_lra_listops_classification_dataset,
	"aan-classification": create_lra_aan_classification_dataset,
	"lra-cifar-classification": create_lra_image_classification_dataset,
	"pathfinder-classification": create_lra_path32_classification_dataset,
	"pathx-classification": create_lra_pathx_classification_dataset,

	# Speech.
	"speech35-classification": create_speechcommands35_classification_dataset,
}

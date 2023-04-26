import torch
from pathlib import Path
from typing import Callable, Optional, TypeVar, Dict, Tuple

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



def create_icl_datasets(config: dict): # -> ReturnType:
	"""

	:param config:		(dict):		config
	:return:
	"""
	print("[*] Generating ICL Dataset")
	from src.dataloaders.synthetics import ICLDataModule

	dataset_obj = ICLDataModule(config.num_examples,
								config.num_test_examples,
								config.vocab_size,
								config.input_seq_len,
								config.copy_method,
								**config.data_kwargs)
	# dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()
	trainloader = dataset_obj.train_dataloader()
	valloader = dataset_obj.val_dataloader()
	testloader = dataset_obj.test_dataloader()

	return trainloader, valloader, testloader


def create_wikitext_dataset(config: dict): # -> ReturnType:
	"""

	:param config:		(dict):		config
	:return:
	"""
	print("[*] Creating wikitext-103 Dataset")
	from src.dataloaders.language_modeling_hf import LMDataModuleWT103

	dataset_obj = LMDataModuleWT103("wikitext",
									"gpt2",
									dataset_config_name="wikitext-103-raw-v1",
									max_length=1024,
                 	cache_dir=config.data_kwargs["data_dir"],
									val_ratio=0.0005,
									val_split_seed=2357,
									add_eos=True,
									detokenize=False,
									val_only=False,
									batch_size=config.data_kwargs["batch_size"],
									batch_size_eval=config.data_kwargs["batch_size_eval"],
									num_workers=config.data_kwargs["num_workers"],
                 	shuffle=True,
									pin_memory=config.data_kwargs["pin_memory"],
									drop_last=False,
									fault_tolerant=False,
									ddp=False,
                 	fast_forward_epochs=None,
									fast_forward_batches=None,
                 	use_shmem=True)
	# dataset_obj.cache_dir = Path(cache_dir) / name
	dataset_obj.setup()
	trainloader = dataset_obj.train_dataloader()
	valloader = dataset_obj.val_dataloader()
	testloader = dataset_obj.test_dataloader()

	return trainloader, valloader, testloader

Datasets = {
	"icl_datasets": create_icl_datasets,
	"wikitext_datasets": create_wikitext_dataset
}

"""Torch DataLoader construction shared by the LOB datasets.

The S5 upstream benchmark loaders (LRA, MNIST, CIFAR, Speech Commands) that
used to live in this module had no caller in sigma-0 and were removed.
"""
import torch
from pathlib import Path
import os
from typing import Callable, Optional, TypeVar, Dict, Tuple, List, Union
from torch.utils.data import Sampler

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
					 collate_fn: callable=None,
					 sampler: Optional[Sampler]=None,
					 num_workers: int = 0,
					 worker_init_fn: Callable = None,
					 pin_memory: bool = True,
					 prefetch_factor: int = 2,
					 persistent_workers: bool = False):
	"""

	:param dset: 			(PT dset):		PyTorch dataset object.
	:param dobj (=None): 	(AG data): 		Dataset object, as returned by A.G.s dataloader.
	:param seed: 			(int):			Int for seeding shuffle.
	:param batch_size: 		(int):			Batch size for batches.
	:param shuffle:         (bool):			Shuffle the data loader?
	:param drop_last: 		(bool):			Drop ragged final batch (particularly for training).
	:param pin_memory:      (bool):			Use pinned memory for faster GPU transfer.
	:param prefetch_factor: (int):			Number of batches to prefetch per worker.
	:param persistent_workers: (bool):		Keep workers alive between epochs.
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

	if sampler is not None:
		shuffle = False

	# Build loader kwargs - some options only valid when num_workers > 0
	loader_kwargs = {
		'dataset': dset,
		'collate_fn': collate_fn,
		'batch_size': batch_size,
		'shuffle': shuffle,
		'drop_last': drop_last,
		'generator': rng,
		'sampler': sampler,
		'num_workers': num_workers,
		'worker_init_fn': worker_init_fn,
		'pin_memory': pin_memory and num_workers > 0,
	}

	if num_workers > 0:
		loader_kwargs['prefetch_factor'] = prefetch_factor
		loader_kwargs['persistent_workers'] = persistent_workers

	return torch.utils.data.DataLoader(**loader_kwargs)

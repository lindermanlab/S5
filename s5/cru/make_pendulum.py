# Code adapted from Schirmer et al [2022, ICML].
# https://github.com/boschresearch/Continuous-Recurrent-Units
# Originally distributed under the GNU Affero General Public License.
# Copyright (c) 2022 Robert Bosch GmbH

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import wandb


# Modified from CRU.
def subsample(data, sample_rate, imagepred=False, random_state=0):
	assert not imagepred, "Imagepred not currently supported."

	train_obs, train_targets = data["train_obs"],  data["train_targets"]
	test_obs, test_targets = data["test_obs"], data["test_targets"]
	valid_obs, valid_targets = data["valid_obs"],  data["valid_targets"]

	seq_length = train_obs.shape[1]
	train_time_points = []
	test_time_points = []
	valid_time_points = []
	n = int(sample_rate*seq_length)

	data_components = train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets
	train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub, valid_obs_sub, valid_targets_sub = \
		[np.zeros_like(x[:, :n, ...]) for x in data_components]

	for i in range(train_obs.shape[0]):
		rng_train = np.random.default_rng(random_state+i+train_obs.shape[0])
		choice = np.sort(rng_train.choice(seq_length, n, replace=False))
		train_time_points.append(choice)
		train_obs_sub[i, ...], train_targets_sub[i, ...] = [
			x[i, choice, ...] for x in [train_obs, train_targets]]

	for i in range(test_obs.shape[0]):
		rng_test = np.random.default_rng(random_state+i)
		choice = np.sort(rng_test.choice(seq_length, n, replace=False))
		test_time_points.append(choice)
		test_obs_sub[i, ...], test_targets_sub[i, ...] = [
			x[i, choice, ...] for x in [test_obs, test_targets]]

	for i in range(valid_obs.shape[0]):
		rng_val = np.random.default_rng(random_state+i+valid_obs.shape[0])
		choice = np.sort(rng_val.choice(seq_length, n, replace=False))
		valid_time_points.append(choice)
		valid_obs_sub[i, ...], valid_targets_sub[i, ...] = [
			x[i, choice, ...] for x in [valid_obs, valid_targets]]

	train_time_points, test_time_points, valid_time_points = \
		np.stack(train_time_points, 0), np.stack(test_time_points, 0), np.stack(valid_time_points, 0)

	return train_obs_sub, train_targets_sub, \
		   test_obs_sub, test_targets_sub, \
		   train_time_points, test_time_points, \
		   valid_obs_sub, valid_targets_sub, valid_time_points


# Modified from CRU.
class Pendulum_regression(Dataset):
	def __init__(self, file_path, name, mode, sample_rate=0.5, random_state=0):

		data = dict(np.load(os.path.join(file_path, name)))

		subsampled = subsample(data, sample_rate=sample_rate, random_state=random_state)
		train_obs, train_targets, test_obs, test_targets, train_time_points, test_time_points, valid_obs, valid_targets, valid_time_points = subsampled

		if mode == 'train':
			self.obs = train_obs
			self.targets = train_targets
			self.time_points = train_time_points
		elif mode == 'test':
			self.obs = test_obs
			self.targets = test_targets
			self.time_points = test_time_points
		elif mode == 'valid':
			self.obs = valid_obs
			self.targets = valid_targets
			self.time_points = valid_time_points
		else:
			raise RuntimeError()

		self.obs = np.ascontiguousarray(
			np.transpose(self.obs, [0, 1, 4, 2, 3]))/255.0

	def __len__(self):
		return self.obs.shape[0]

	def __getitem__(self, idx):
		obs = torch.from_numpy(self.obs[idx, ...].astype(np.float64))
		targets = torch.from_numpy(self.targets[idx, ...].astype(np.float64))
		time_points = torch.from_numpy(self.time_points[idx, ...])
		obs_valid = torch.ones_like(time_points, dtype=torch.bool)
		return obs, targets, time_points, obs_valid


# Modified from CRU.
def load_pendulum_regression_data(args):
	file_path = f'{args.cache_dir}/{args.dataset}/'
	file_name = f'pend_regression.npz'

	if not os.path.exists(os.path.join(file_path, file_name)):
		raise NotImplementedError("Dataset not found.  EITHER `dir_name` is incorrect, or, the "
								  "dataset has not been generated.  Please check the `dir_name` "
								  "argument.  If the argument is correct (nearly always `./cache_dir`) "
								  "then it is most likely that the dataset has not been generated.  "
								  "Please use the code and instructions distributed at "
								  "https://github.com/andrewwarrington/Continuous-Recurrent-Units "
								  "to generate the data.")
	else:
		print(f'Loading from {file_path}.')

	renamed_args = {'file_path': file_path,
					'name': file_name,
					'sample_rate': args.sample_rate,
					'random_state': args.data_random_seed, }

	train = Pendulum_regression(mode='train', **renamed_args)
	test = Pendulum_regression(mode='test', **renamed_args)
	val = Pendulum_regression(mode='valid', **renamed_args)

	# Compute some checksums for sanity between CRU and S5.
	print('\nSanity Checksums:')
	chksum_dict = {'chksum/obs/train': train.obs.sum(),
				   'chksum/obs/test': test.obs.sum(),
				   'chksum/obs/val': val.obs.sum(),
				   'chksum/targets/train': train.targets.sum(),
				   'chksum/targets/test': test.targets.sum(),
				   'chksum/targets/val': val.targets.sum(),
				   'chksum/time_points/train': train.time_points.sum(),
				   'chksum/time_points/test': test.time_points.sum(),
				   'chksum/time_points/val': val.time_points.sum()}
	wandb.log(chksum_dict, commit=False)
	print(chksum_dict)
	print('Train targets shape: ', train.targets.shape)
	print('val targets shape:   ', val.targets.shape)
	print('Test targets shape:  ', test.targets.shape)

	return train, test, val

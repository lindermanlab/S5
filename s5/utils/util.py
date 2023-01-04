import argparse
from typing import Sequence
from flax import linen as nn


class SimpleMLP(nn.Module):
	features: Sequence[int]

	@nn.compact
	def __call__(self, inputs):
		x = inputs
		for i, feat in enumerate(self.features):
			x = nn.Dense(feat, name=f'layers_{i}')(x)
			if i != len(self.features) - 1:
				x = nn.tanh(x)
			# providing a name is optional though!
			# the default autonames would be "Dense_0", "Dense_1", ...
		return x


def is_list(x):
	"""
	From AG:  this is usually used in a pattern where it's turned into a list, so can just do that here
	:param x:
	:return:
	"""
	return isinstance(x, Sequence) and not isinstance(x, str)


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

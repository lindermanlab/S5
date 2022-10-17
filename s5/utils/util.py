from typing import Sequence
import argparse


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

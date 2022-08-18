from typing import Sequence, Mapping, Optional, Callable


# TODO this is usually used in a pattern where it's turned into a list, so can just do that here
def is_list(x):
	return isinstance(x, Sequence) and not isinstance(x, str)


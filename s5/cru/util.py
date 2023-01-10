import jax
import jax.numpy as np
from flax import linen as nn


class CRN_CNN(nn.Module):

	d_output: int
	input_shape: int
	append_timestep: bool = False

	@nn.compact
	# Provide a constructor to register a new parameter
	# and return its initial value
	def __call__(self, x, _integration_timesteps=None):
		x = x.reshape(x.shape[0], self.input_shape, self.input_shape, 1)
		x = nn.Conv(features=12, kernel_size=(5, 5), padding=((2, 2), (2, 2)))(x)
		x = nn.relu(x)
		x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = nn.Conv(features=12, kernel_size=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))(x)
		x = nn.relu(x)
		x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
		x = x.reshape((x.shape[0], -1))  # Flatten

		x = nn.Dense(features=30)(x)
		x = nn.relu(x)

		if self.append_timestep:
			print('\n\nWarning:  appending integration timesteps. \n\n')
			# TODO - fix for with bidirectional.
			_integration_timesteps = np.expand_dims(np.concatenate((np.asarray([1.0]), _integration_timesteps)), axis=1)
			x = np.concatenate((x, _integration_timesteps), axis=-1)

		# This is in place of the linear output for latent observation mean/variance.
		x = nn.Dense(features=self.d_output)(x)
		return x

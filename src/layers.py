from flax import linen as nn


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_scale  (float32):  allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_scale: float = 1.0

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """
        self.seq = self.ssm(step_scale=self.step_scale)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.

        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)

        """
        skip = x
        if self.prenorm:
            x = self.norm(x)
        x = self.seq(x)
        x = self.drop(nn.gelu(x))
        x = skip + x
        if not self.prenorm:
            x = self.norm(x)
        return x

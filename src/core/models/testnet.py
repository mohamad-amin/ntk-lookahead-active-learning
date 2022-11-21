import torch
import torch.nn as nn
import torch.nn.functional as F

import jax
from flax import linen
from jax import numpy as jnp
from src.core.models.wide_resnet_jax import Sequential

PRECISION = jax.lax.Precision(2)  # 0: 16bit - 1: 32bit - 2: 64bit
DTYPE = jnp.float64


class TestNet(nn.Module):

    def __init__(self, num_layers, depth, widen_factor, dropout_rate, num_classes,
                 num_input_channels=3, norm_layer=None):
        super(TestNet, self).__init__()
        self.layer1 = nn.Linear(10, 10 * widen_factor)
        self.layer2 = nn.Linear(10 * widen_factor, 3)

    def forward(self, input_dict):
        x = self.layer1(input_dict['inputs'])
        x = self.layer2(x)
        return x


def TestNetNTK(num_layers, depth, widen_factor, dropout_rate, num_classes,
                 num_input_channels=3, norm_layer=None):

    layer1 = linen.Dense(10 * widen_factor, dtype=DTYPE, precision=PRECISION)
    layer2 = linen.Dense(3, dtype=DTYPE, precision=PRECISION)
    return Sequential([layer1, layer2])

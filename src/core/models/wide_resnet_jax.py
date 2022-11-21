from functools import partial
from typing import (Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple,
                    Union)

import jax
from jax import lax
import jax.numpy as jnp
from flax import linen as nn


ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]

PRECISION = jax.lax.Precision(2)  # 0: 16bit - 1: 32bit - 2: 64bit
DTYPE = jnp.float64


STAGE_SIZES = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
    269: [3, 30, 48, 8],
}


class ConvBlock(nn.Module):
    n_filters: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    activation: Callable = nn.relu
    padding: Union[str, Iterable[Tuple[int, int]]] = ((0, 0), (0, 0))
    is_last: bool = False
    groups: int = 1
    kernel_init: InitFn = nn.initializers.kaiming_normal()
    bias_init: InitFn = nn.initializers.zeros

    conv_cls: ModuleDef = partial(nn.Conv, dtype=DTYPE, precision=PRECISION)
    norm_cls: Optional[ModuleDef] = partial(nn.BatchNorm, momentum=1.0, dtype=DTYPE)

    force_conv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        x = self.conv_cls(
            self.n_filters,
            self.kernel_size,
            self.strides,
            use_bias=(not self.norm_cls or self.force_conv_bias),
            padding=self.padding,
            feature_group_count=self.groups,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        if self.norm_cls:
            scale_init = (nn.initializers.zeros
                          if self.is_last else nn.initializers.ones)
            x = self.norm_cls(use_running_average=True, scale_init=scale_init)(x)

        if not self.is_last:
            x = self.activation(x)
        return x


class Sequential(nn.Module):
    layers: Sequence[Union[nn.Module, Callable[[jnp.ndarray], jnp.ndarray]]]

    @nn.compact
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class WideBasicNtk(nn.Module):
    in_planes: int
    planes: int
    stride: int

    def setup(self):
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=1.0, dtype=DTYPE)
        self.conv1 = nn.Conv(self.planes, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)], dtype=DTYPE, precision=PRECISION)
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=1.0, dtype=DTYPE)
        self.conv2 = nn.Conv(self.planes, (3, 3), strides=(self.stride, self.stride), padding=[(1, 1), (1, 1)], dtype=DTYPE, precision=PRECISION)
        self.shortcut = None
        if self.stride != 1 or self.in_planes != self.planes:
            self.shortcut = nn.Conv(self.planes, (1, 1), strides=(self.stride, self.stride), padding=[(0, 0), (0, 0)], dtype=DTYPE,
                               precision=PRECISION)

    def __call__(self, inp):
        x = self.conv1(nn.relu(self.bn1(inp)))
        # x = self.conv1(inp)
        x = self.conv2(nn.relu(self.bn2(x)))
        # x = self.conv2(x)
        if self.shortcut is not None:
            x += self.shortcut(inp)
        return x


def WideResNetNTK(
    num_layers: int,
    widen_factor: int,
    depth: int,
    num_classes: int,
    num_input_channels: int = 3,
    dropout_rate: float = 0.3) -> Sequential:

    assert ((depth-4)%6 ==0), 'wide-resnet depth should be 6n+4'
    n = (depth-4)/6
    k = widen_factor

    def _wide_layer(block, in_planes, planes, num_blocks, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(in_planes, planes, stride))
            in_planes = planes

        return Sequential(layers), in_planes

    print('| wide-resnet %dx%d' %(depth, k))
    nstages = [16*k, 32*k, 64*k, 128*k]
    in_planes = nstages[0]

    conv1 = nn.Conv(nstages[0], (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)], dtype=DTYPE, precision=PRECISION)
    if num_layers >= 1:
        layer1, in_planes = _wide_layer(WideBasicNtk, in_planes, nstages[1],
                                        n, stride=1)
    if num_layers >= 2:
        layer2, in_planes = _wide_layer(WideBasicNtk, in_planes, nstages[2],
                                        n, stride=2)
    if num_layers == 3:
        layer3, in_planes = _wide_layer(WideBasicNtk, in_planes, nstages[3],
                                        n, stride=2)
    bn1 = nn.BatchNorm(use_running_average=True, momentum=1.0, dtype=DTYPE)
    linear = nn.Dense(num_classes, dtype=DTYPE, precision=PRECISION)
    avg_pool = partial(jnp.mean, axis=(1, 2))

    if num_layers == 1:
        net = Sequential([
            conv1,
            layer1,
            bn1,
            nn.relu,
            avg_pool,
            linear
        ])
    elif num_layers == 2:
        net = Sequential([
            conv1,
            layer1, layer2,
            bn1,
            nn.relu,
            avg_pool,
            linear
        ])
    elif num_layers == 3:
        net = Sequential([
            conv1,
            layer1, layer2, layer3,
            bn1,
            nn.relu,
            avg_pool,
            linear
        ])

    return net

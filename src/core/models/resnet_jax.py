from functools import partial
from typing import (Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple,
                    Union, Sized, List)

import jax
from jax import lax
import jax.numpy as jnp
from flax import linen as nn

from src.core.models.wide_resnet_jax import Sequential

ModuleDef = Callable[..., Callable]
InitFn = Callable[[Any, Iterable[int], Any], Any]

PRECISION = jax.lax.Precision(2)  # 0: 16bit - 1: 32bit - 2: 64bit
DTYPE = jnp.float64

PreciseConv = partial(nn.Conv, dtype=DTYPE, precision=PRECISION)
PreciseBatchNorm = partial(nn.BatchNorm, use_running_average=True, momentum=1.0, dtype=DTYPE)


class BasicBlock(nn.Module):
    in_planes: int
    planes: int
    stride: int = 1
    expansion = 1

    def setup(self):
        self.conv1 = PreciseConv(self.planes, (3, 3),
                                 strides=(self.stride, self.stride), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn1 = PreciseBatchNorm()
        self.conv2 = PreciseConv(self.planes, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn2 = PreciseBatchNorm()

        shortcut = Sequential([])
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            p = self.expansion * self.planes
            shortcut = Sequential([
                PreciseConv(p, (1, 1), strides=(self.stride, self.stride), padding=[(0, 0), (0, 0)], use_bias=False),
                PreciseBatchNorm()
            ])
        self.shortcut = shortcut

    def __call__(self, inp):
        out = nn.relu(self.bn1(self.conv1(inp)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inp)
        out = nn.relu(out)
        return out


class Bottleneck(nn.Module):
    in_planes: int
    planes: int
    stride: int = 1
    expansion: int = 4

    def setup(self):
        self.conv1 = PreciseConv(self.planes, (1, 1),
                                 strides=(self.stride, self.stride), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn1 = PreciseBatchNorm()
        self.conv2 = PreciseConv(self.planes, (3, 3),
                                 strides=(self.stride, self.stride), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn2 = PreciseBatchNorm()
        self.conv3 = PreciseConv(self.expansion * self.planes, (1, 1),
                                 strides=(1, 1), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn3 = PreciseBatchNorm()

        shortcut = Sequential([])
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            shortcut = Sequential([
                PreciseConv(self.expansion * self.planes, (1, 1),
                            strides=(self.stride, self.stride), padding=[(0, 0), (0, 0)], use_bias=False),
                PreciseBatchNorm()
            ])
        self.shortcut = shortcut

    def __call__(self, inp):
        out = nn.relu(self.bn1(self.conv1(inp)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(inp)
        out = nn.relu(out)
        return out


class ResNet(nn.Module):
    block: nn.Module
    num_blocks: List
    num_classes: int = 10
    num_input_channels: int = 3

    def setup(self):
        self.in_planes = 64
        self.num_layers = len(self.num_blocks)

        self.conv1 = PreciseConv(64, (3, 3), strides=(1, 1), padding=[(1, 1), (1, 1)], use_bias=False)
        self.bn1 = PreciseBatchNorm()
        if self.num_layers >= 1:
            self.layer1 = self._make_layer(self.block, 64, self.num_blocks[0], stride=1)
        if self.num_layers >= 2:
            self.layer2 = self._make_layer(self.block, 128, self.num_blocks[1], stride=2)
        if self.num_layers >= 3:
            self.layer3 = self._make_layer(self.block, 256, self.num_blocks[2], stride=2)
        if self.num_layers >= 4:
            self.layer4 = self._make_layer(self.block, 512, self.num_blocks[3], stride=2)
        self.linear = nn.Dense(self.num_classes, dtype=DTYPE, precision=PRECISION)
        self.avgpool = partial(jnp.mean, axis=(1, 2))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(layers)

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        if self.num_layers == 1:
            out1 = self.layer1(out)
            out = self.avgpool(out1)
            emb = out.reshape(out.shape[0], -1)  # flatten
            out = self.linear(emb)
        elif self.num_layers == 2:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out = self.avgpool(out2)
            emb = out.reshape(out.shape[0], -1)  # flatten
            out = self.linear(emb)
        elif self.num_layers == 3:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out = self.avgpool(out3)
            emb = out.reshape(out.shape[0], -1)  # flatten
            out = self.linear(emb)
        else:
            out1 = self.layer1(out)
            out2 = self.layer2(out1)
            out3 = self.layer3(out2)
            out4 = self.layer4(out3)
            out = self.avgpool(out4)
            emb = out.reshape(out.shape[0], -1)  # flatten
            out = self.linear(emb)
        return out


def ResNet18(num_layers, depth, widen_factor, dropout_rate, num_classes,
                 num_input_channels=3, norm_layer=None):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, num_input_channels)


def ResNet34(num_layers, depth, widen_factor, dropout_rate, num_classes,
                 num_input_channels=3, norm_layer=None):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, num_input_channels)


def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])


def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])


def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

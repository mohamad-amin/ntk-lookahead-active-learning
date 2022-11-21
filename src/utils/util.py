import torch
import getpass
import logging
import os
import re
import yaml
import time
import numpy as np
from colorlog import ColoredFormatter
import jax
from jax import tree_util
from jax.ops import index, index_update
from scipy.special import log_softmax
from neural_tangents import predict


# Logging
# =======

def load_log(name):
    def _infov(self, msg, *args, **kwargs):
        self.log(logging.INFO + 1, msg, *args, **kwargs)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s - %(name)s] %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'white,bold',
            'INFOV':    'cyan,bold',
            'WARNING':  'yellow',
            'ERROR':    'red,bold',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    ch.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.handlers = []       # No duplicated handlers
    log.propagate = False   # workaround for duplicated logs in ipython
    log.addHandler(ch)

    logging.addLevelName(logging.INFO + 1, 'INFOV')
    logging.Logger.infov = _infov
    return log


# General utils
# =============

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# Path utils
# ==========

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)
    return path


# MultiGPU
# ========

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# Data
# ====

def to_device(dict, device):
    for key in dict.keys():
        if isinstance(dict[key], torch.Tensor):
            dict[key] = dict[key].to(device)
    return dict


# Divisors of a number
# ====================

def get_sorted_divisors(num):
    candidates = np.arange(1, num+1)
    return candidates[np.mod(num, candidates) == 0]


# Get params size of jax neural network
# =====================================

def get_params_size(params):
    return sum(x.size for x in jax.tree_leaves(params))

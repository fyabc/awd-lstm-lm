#! /usr/bin/python
# -*- coding: utf-8 -*-

import contextlib
import random
import subprocess

import numpy as np
import torch as th
from torch.autograd import Variable

__author__ = 'fyabc'


def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.random.manual_seed(seed)


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


def volatile_variable(*args, **kwargs):
    if hasattr(th, 'no_grad'):
        # volatile has been deprecated, use the no_grad context manager instead
        return Variable(*args, **kwargs)
    else:
        return Variable(*args, **kwargs, volatile=True)


def make_variable(sample, volatile=False, cuda=False, requires_grad=False):
    """Wrap input tensors in Variable class."""

    def _make_variable(maybe_tensor):
        if th.is_tensor(maybe_tensor):
            if cuda and th.cuda.is_available():
                maybe_tensor = maybe_tensor.cuda()
            if volatile:
                return volatile_variable(maybe_tensor, requires_grad=requires_grad)
            else:
                return Variable(maybe_tensor, requires_grad=requires_grad)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _make_variable(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_make_variable(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _make_variable(sample)


def make_int_scalar_variable(x, cuda=False):
    return make_variable(th.ones([]).long().fill_(x), cuda=cuda)


def num_parameters(module):
    return sum(p.numel() for p in module.parameters())


def get_gpu_memory_map():
    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ])
        return float(result)
    except OSError:
        import math
        return math.nan


@contextlib.contextmanager
def hparams_env(hparams, **kwargs):
    old_dict = {k: getattr(hparams, k) for k in kwargs}
    hparams.__dict__.update(kwargs)
    try:
        yield old_dict
    finally:
        hparams.__dict__.update(old_dict)

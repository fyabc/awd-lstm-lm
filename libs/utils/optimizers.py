#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch


def build_optimizer(args, name, params, prefix=''):
    def _get(key):
        return getattr(args, '{}{}'.format(prefix, key))

    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's
    # weight (i.e. Adaptive Softmax)
    if name == 'sgd':
        return torch.optim.SGD(params, lr=_get('lr'), weight_decay=_get('wdecay'))
    elif name == 'adam':
        return torch.optim.Adam(params, lr=_get('lr'), weight_decay=_get('wdecay'))
    else:
        raise NotImplementedError('Optimizer {!r} not supported now'.format(name))

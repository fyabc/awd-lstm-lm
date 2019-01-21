#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch


def cat(xs):
    return torch.cat([x.view(-1) for x in xs])


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


class CycledBatchIterator:
    def __init__(self, args, data_source):
        self.args = args
        self.data_source = data_source
        self._seq_len = None
        self._loop_range = self._get_range()

    def _get_range(self):
        return iter(range(0, self.data_source.size(0) - 1, self.args.teacher_bptt))

    @property
    def seq_len(self):
        return self._seq_len

    @seq_len.setter
    def seq_len(self, value):
        self._seq_len = value

    def __iter__(self):
        return self

    def __next__(self):
        try:
            i = next(self._loop_range)
        except StopIteration:
            self._loop_range = self._get_range()
            i = next(self._loop_range)
        return get_batch(self.data_source, i, self.args, seq_len=self.seq_len, evaluation=True)


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer, lr):
    optimizer.param_groups[0]['lr'] = lr

#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import math
import pickle
import weakref

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..utils import common
from ..utils.paths import CdfPath
from .embed_regularize import embedded_dropout


class BaseFeatureExtractor(nn.Module):
    def __init__(self, hparams, task, student):
        super().__init__()
        self.hparams = hparams

        self.epoch_embedding = nn.Embedding(hparams.max_student_epoch, hparams.epoch_emb_size)

        # Accuracy embedding.
        self.acc_embedding = None
        self._acc_buckets = None
        if hparams.enable_acc_emb:
            self.acc_embedding = nn.Embedding(hparams.n_acc_buckets, hparams.acc_emb_size)
            self._acc_buckets = self._build_acc_buckets()
            if self._acc_buckets is not None:
                self._total_len = len(self._acc_buckets)

        total_emb_size = hparams.data_feature_size + hparams.epoch_emb_size
        if self.use_acc_embedding:
            total_emb_size += hparams.acc_emb_size
        self.output_shape = th.Size([1, total_emb_size])

    def _build_acc_buckets(self):
        full_series_name = '{}-{:.1f}'.format(self.hparams.dataset, getattr(self.hparams, 'train_size', 1.0))
        baseline_n_layers = len(self.hparams.model_space)   # FIXME: Hard code here.
        with open(CdfPath, 'rb') as f:
            data = pickle.load(f)
            try:
                return data[full_series_name][baseline_n_layers]
            except KeyError:
                print('Key {!r} in CDF file does not exists, CDF accuracy embedding disabled!'.format(full_series_name))
                return None

    def _get_acc_bucket(self, dev_acc):
        dev_acc_idx = int(np.searchsorted(self._acc_buckets, dev_acc) / self._total_len * self.hparams.n_acc_buckets)
        if dev_acc_idx >= self.hparams.n_acc_buckets:
            dev_acc_idx -= 1
        return common.make_int_scalar_variable(dev_acc_idx, cuda=self.hparams.cuda)

    @property
    def use_acc_embedding(self):
        return self._acc_buckets is not None

    def forward(self, sample, epoch, child_model=None, dev_acc=None):
        """

        Args:
            sample (dict): The input sample.
            epoch: The scalar tensor of epoch number.
            child_model: The student model or None.
            dev_acc: The accuracy on dev set.

        Returns:
            Final embedding
        """

        batch_size = self._get_batch_size(sample)

        data_emb = self._get_data_embedding(sample)

        epoch_emb = self.epoch_embedding(epoch)
        to_be_concated = [
            data_emb,
            epoch_emb.unsqueeze(0).repeat(batch_size, 1),
        ]

        if dev_acc is not None and self.use_acc_embedding:
            acc_emb = self.acc_embedding(self._get_acc_bucket(dev_acc))
            to_be_concated.append(acc_emb.unsqueeze(0).repeat(batch_size, 1))

        final_emb = th.cat(to_be_concated, dim=-1)
        return final_emb

    def _get_data_embedding(self, sample):
        raise NotImplementedError

    def _get_batch_size(self, sample):
        raise NotImplementedError


class WT2FeatureExtractor(BaseFeatureExtractor):
    # TODO

    def __init__(self, hparams, task, student):
        super().__init__(hparams, task, student)

        self.num_layers = 1
        self.bidirectional = False

        self._build_data_embedding(student)

    def _build_data_embedding(self, student):
        self.student = weakref.ref(student)
        self.input_shape = th.Size([1, 1, self.hparams.emsize])
        self.lstm = nn.LSTM(
            input_size=self.input_shape[-1],
            hidden_size=self.hparams.data_feature_size,
            num_layers=self.num_layers,
            batch_first=False,
            bidirectional=self.bidirectional,
        )

    def _get_data_embedding(self, sample):
        data = sample['data']
        bsz = self._get_batch_size(sample)

        # # FIXME: disable dropout in teacher or not?
        # emb = embedded_dropout(
        #     self.student().encoder, data,
        #     dropout=0,
        # )
        emb = embedded_dropout(
            self.student().encoder, data,
            dropout=self.hparams.dropoute if self.training else 0,
        )
        emb = self.student().lockdrop(emb, self.student().dropouti)

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hparams.data_feature_size
        else:
            state_size = self.num_layers, bsz, self.hparams.data_feature_size
        x = emb
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        output, (_, _) = self.lstm(x, (h0, c0))

        # average over time to get data embedding
        return output.mean(dim=0)

    def _get_batch_size(self, sample):
        return sample['data'].size(1)


class Teacher(nn.Module):
    """The teacher model.

    Contains:
        feature extractors
            data feature extractor (a small and simple CNN)
            model feature extractor (dev loss/accuracy embedding)
            training feature extractor (epoch number embedding) (optional)

        shared body

        multi-task output
            model arch predictor
            loss structure predictor
            data selection predictor (hard / soft selection)
    """

    DisableGS = defaultdict(lambda: False)

    def __init__(self, hparams, task, student):
        super().__init__()
        self.hparams = hparams

        self.feature_extractor = WT2FeatureExtractor(hparams, task, student)

        self._build_body(self.feature_extractor.output_shape)

        self.model_predictor = nn.Linear(hparams.body_hidden_size, len(hparams.model_space))

        if hparams.encoder_model_space is not None:
            self.encoder_model_predictor = nn.Linear(hparams.body_hidden_size, len(hparams.encoder_model_space))
        else:
            self.encoder_model_predictor = None

        loss_space_size = len(hparams.loss_space)
        if loss_space_size == 1:
            self.loss_predictor = None
        else:
            self.loss_predictor = nn.Linear(hparams.body_hidden_size, loss_space_size)

        if hparams.no_data_selection:
            self.data_predictor = None
        else:
            # [0 = False, 1 = True] two options.
            # Also used as soft: value[1] indicate the weight.
            self.data_predictor = nn.Linear(hparams.body_hidden_size, 2)

        self._init_weights()

    @classmethod
    def build_teacher(cls, args, model, state_dict=None):
        # Build the teacher
        teacher = cls(args, task=None, student=model)
        if state_dict is not None:
            teacher.load_state_dict(state_dict)
        if args.cuda:
            teacher = teacher.cuda()
        return teacher

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _build_body(self, input_shape):
        batch_size, feature_size = input_shape
        hidden_size = self.hparams.body_hidden_size

        mlp_layers = []

        for i in range(self.hparams.body_num_layers):
            in_size = feature_size if i == 0 else hidden_size
            mlp_layers.append(nn.Linear(in_size, hidden_size))
        self.body = nn.Sequential(*mlp_layers)

    def forward(self, sample, epoch, child_model, dev_acc=None):
        feature = self.feature_extractor(sample, epoch, child_model, dev_acc=dev_acc)

        body_out = self.body(feature)

        # [NOTE]: All are before softmax, remain unnormalized
        model_out = self.model_predictor(body_out)
        loss_out = self.loss_predictor(body_out) if self.loss_predictor is not None else None

        if self.data_predictor is None:
            data_out = None
        else:
            data_out = self.data_predictor(body_out)

        if self.encoder_model_predictor is None:
            encoder_model_out = None
        else:
            encoder_model_out = self.encoder_model_predictor(body_out)

        return {
            'model_out': model_out,
            'encoder_model_out': encoder_model_out,
            'loss_out': loss_out,
            'data_out': data_out,
        }

    def teacher_selection_step(self, sample, epoch, train=None):
        """

        Args:
            sample (dict):
                data: (seq_len, batch_size)
                targets: (seq_len * batch_size)
            epoch (int):
            train:

        Returns:

        """
        # FIXME: epsilon greedy disabled now.

        if sample is None:
            return {}
        if train is not None:
            self.train(mode=train)
        epoch_var = common.make_int_scalar_variable(epoch - 1, cuda=self.hparams.cuda)
        teacher_out = self(sample, epoch_var, child_model=None, dev_acc=None)

        # FIXME: Temperature scheduler disabled now.
        gs_tau = 1.0

        data_weight = self._get_data_selection(teacher_out['data_out'], gs_tau)
        loss_selection = self._get_selection('loss', teacher_out['loss_out'], gs_tau)
        model_selection = self._get_selection('model', teacher_out['model_out'], gs_tau)
        encoder_model_selection = self._get_selection('encoder_model', teacher_out['encoder_model_out'], gs_tau)

        return {
            'data_weight': data_weight,
            'loss_selection': loss_selection,
            'model_selection': model_selection,
            'encoder_model_selection': encoder_model_selection,
        }

    def _get_data_selection(self, data_out, gs_tau):
        if data_out is None:
            return None
        if self.DisableGS['data']:
            return F.softmax(data_out, dim=1)
        softmax = F.gumbel_softmax(
            data_out, tau=gs_tau, hard=self.hparams.hard_data_selection, eps=self.hparams.gs_eps)
        # FIXME: Apply "soft" epsilon-greedy selection, or not apply selection?
        # softmax = self._epsilon_greedy_wrapper(softmax, hard=False, apply=apply_eg)
        return softmax[:, 1:].squeeze()

    def _get_selection(self, key, out, gs_tau, hard=True):
        if out is None:
            # Only one candidate, no selection.
            return None
        if self.DisableGS[key]:
            return F.softmax(out, dim=1)
        out = F.gumbel_softmax(out, tau=gs_tau, hard=hard, eps=self.hparams.gs_eps)
        return out

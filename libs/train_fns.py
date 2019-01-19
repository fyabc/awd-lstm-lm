#! /usr/bin/python
# -*- coding: utf-8 -*-


def prepare_encoder_kw(hparams, teacher_out):
    if teacher_out is None:
        kw = {'model_selection': hparams.model_focus}
        if hparams.encoder_model_space is not None:
            kw['encoder_model_selection'] = hparams.encoder_model_focus
    else:
        kw = {'model_selection': teacher_out['model_selection']}
        if hparams.encoder_model_space is not None:
            kw['encoder_model_selection'] = teacher_out['encoder_model_selection']
    return kw


def student_forward(hparams, teacher, student, sample, teacher_out, criterion):
    teacher.train()
    student.train()

    data, hidden, targets = sample['data'], sample['hidden'], sample['targets']

    output, hidden, rnn_hs, dropped_rnn_hs = student_out = student(
        data, hidden, return_h=True, **prepare_encoder_kw(hparams, teacher_out))

    # TODO: Change the raw loss into L2TE loss.
    raw_loss = criterion(student.decoder.weight, student.decoder.bias, output, targets)

    return student_out, raw_loss

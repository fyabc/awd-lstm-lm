#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import torch as th


from .utils.utils import cat, get_lr


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
    loss = student.loss_regularization(hparams, raw_loss, rnn_hs, dropped_rnn_hs)

    return student_out, raw_loss, loss


def whole_loss(trainer, teacher, student, sample, epoch):
    teacher_out = teacher.teacher_selection_step(sample, epoch, train=None)
    student_out, raw_loss, loss = student_forward(
        trainer.hparams, teacher, student, sample, teacher_out, trainer.criterion)
    return loss


# Train step (second order) methods.

def teacher_train_step(trainer, sample, epoch, train_sample=None, train=True):
    hparams, teacher, student = trainer.hparams, trainer.teacher, trainer.student

    teacher_out = teacher.teacher_selection_step(sample, epoch, train=train)

    teacher.train(mode=train)
    student.train(mode=train)

    student_out, raw_loss, loss = student_forward(hparams, teacher, student, sample, teacher_out, trainer.criterion)

    objective = loss
    if hparams.first_order:
        objective.backward()
    else:
        assert train_sample is not None, 'Second-order optimization requires the train sample'
        eta = get_lr(trainer.student_optimizer)
        backward_step_unrolled(trainer, sample, train_sample, eta, epoch)

    trainer.teacher_optimizer.step()
    trainer.teacher_optimizer.zero_grad()

    return {
        'student_out': student_out,
        'teacher_objective': objective,
    }


def backward_step_unrolled(trainer, sample, train_sample, eta, epoch):
    """Compute the second-order gradient with unrolled model.

    w = student
    alpha = teacher
    w' = unrolled student
    """

    # 1. Compute unrolled teacher and student
    unrolled_models = _compute_unrolled_model(trainer, train_sample, eta, epoch)
    teacher_u, student_u = unrolled_models['teacher'], unrolled_models['student']

    # 2. Compute the gradients of objective (d{Objective-on-dev(w', alpha)}/d{alpha}) on dev set.
    objective = whole_loss(trainer, teacher_u, student_u, sample, epoch)
    grads = th.autograd.grad(objective, teacher_u.parameters(), retain_graph=True)

    # 3. Modify the gradients with second order gradient approximation:
    # d^2{Loss-train(w, alpha)} / (d{alpha}d{w}) * d{Objective-on-dev(w', alpha)} / d{w'}
    # ~= ``self._hessian_vector_product`` (see doc of this method)
    theta = student_u.parameters()
    dtheta = th.autograd.grad(objective, student_u.parameters())
    # ``vector`` is d{Objective-on-dev(w', alpha)} / d{w'}.
    vector = [dt.add(trainer.hparams.wdecay, t).data for dt, t in zip(dtheta, theta)]
    implicit_grads = _hessian_vector_product(trainer, teacher_u, student_u, vector, train_sample, epoch)

    for g, ig in zip(grads, implicit_grads):
        g.data.sub_(eta, ig.data)

    # 4. Assign the gradient.
    for v, g in zip(trainer.teacher.parameters(), grads):
        if v.grad is None:
            v.grad = th.autograd.Variable(g.data)
        else:
            # [NOTE]: Change `copy_` to `add_`, accumulate gradients.
            v.grad.data.add_(g.data)


def _compute_unrolled_model(trainer, train_sample, eta, epoch):
    """Compute the unrolled student (after a single student SGD step)

    Args:
        trainer:
        train_sample:
        eta: Student learning rate
        epoch:

    Returns:
        dict: Unrolled models
        Keys:
            teacher: Unrolled teacher (same as original teacher)
            student: Unrolled student
    """
    # [NOTE]: Assume that the student optimizer is Nesterov Momentum SGD.

    student_loss = whole_loss(trainer, trainer.teacher, trainer.student, train_sample, epoch)
    theta = cat(trainer.student.parameters()).data
    try:
        moment = cat(
            trainer.student_optimizer.optimizer[v]['momentum_buffer'] for v in trainer.student.parameters()
        ).mul_(trainer.hparams.student_momentum)
    except:
        moment = th.zeros_like(theta)
    dtheta = cat(th.autograd.grad(student_loss, trainer.student.parameters())).data + \
        trainer.hparams.wdecay * theta

    unrolled_models = _construct_from_theta(trainer, theta.sub(eta, moment + dtheta))
    return unrolled_models


def _construct_from_theta(trainer, theta):
    from .modules.teacher import Teacher
    from .modules.model import RNNModel as Student

    hparams, teacher, student = trainer.hparams, trainer.teacher, trainer.student

    student_clone = Student.build_model(hparams)
    if hparams.cuda:
        student_clone.cuda()
    teacher_clone = Teacher.build_teacher(hparams, student_clone, state_dict=teacher.state_dict())

    # Copy all student parameters.
    student_dict = student.state_dict()
    params, offset = {}, 0
    for k, v in student.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset + v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    student_dict.update(params)
    student_clone.load_state_dict(student_dict)

    return {
        'teacher': teacher_clone,
        'student': student_clone,
    }


def _hessian_vector_product(trainer, teacher_u, student_u, vector, train_sample, epoch, r=1e-2):
    """Compute the hessian vector product approximation.

    Second-order gradient ~=
        (d{Loss-train(w+, alpha)} / d{alpha} - d{Loss-train(w-, alpha)} / d{alpha}) / (2 * epsilon)

    Args:
        trainer:
        teacher_u:
        student_u:
        vector:
        train_sample:
        epoch:
        r:

    Returns:

    """
    R = r / cat(vector).norm()
    for p, v in zip(student_u.parameters(), vector):
        p.data.add_(R, v)
    loss = whole_loss(trainer, teacher_u, student_u, train_sample, epoch)
    grads_p = th.autograd.grad(loss, teacher_u.parameters())

    for p, v in zip(student_u.parameters(), vector):
        p.data.sub_(2 * R, v)
    loss = whole_loss(trainer, teacher_u, student_u, train_sample, epoch)
    grads_n = th.autograd.grad(loss, teacher_u.parameters())

    for p, v in zip(student_u.parameters(), vector):
        p.data.add_(R, v)

    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

#! /usr/bin/python
# -*- coding: utf-8 -*-


def add_teacher_args(parser):
    group = parser.add_argument_group('Teacher Options', description='Teacher hparams options.')

    group.add_argument('--episode', type=int, default=1,
                       help='Episode number to train, default is %(default)r')
    group.add_argument('--max-student-epoch', type=int, default=750,
                       help='Max student epoch (used for epoch embedding), default is %(default)s')
    group.add_argument('--epoch-emb-size', type=int, default=32,
                       help='Epoch embedding size, default is %(default)s')
    group.add_argument('--enable-acc-emb', action='store_true', default=False,
                       help='Enable dev accuracy embedding, default is disabled')
    group.add_argument('--n-acc-buckets', type=int, default=20,
                       help='Number of accuracy buckets (#embeddings), default is %(default)r')
    group.add_argument('--acc-emb-size', type=int, default=32,
                       help='Dev accuracy embedding size, default is %(default)r')
    group.add_argument('--data-feature-size', type=int, default=128,
                       help='Data feature size, default is %(default)s')
    group.add_argument('--body-hidden-size', type=int, default=128,
                       help='Teacher shared body hidden size, default is %(default)s')
    group.add_argument('--body-num-layers', type=int, default=2,
                       help='Teacher shared body #layers, default is %(default)s')
    group.add_argument('--model-space', type=str, default='2,1',
                       help='Teacher model space, default is %(default)r, are integers, split by commas')
    group.add_argument('--model-focus', type=int, default=None,
                       help='Focus on which model selection in raw train step.'
                            'Focus on i means student will using i-th linear as output '
                            '(index start at 1, from shallow to deep).'
                            'Default is None (means last)')
    group.add_argument('--encoder-model-space', type=str, default=None,
                       help='Teacher encoder model space, default is %(default)r, are integers, split by commas '
                            '(only used in text tasks)')
    group.add_argument('--encoder-model-focus', type=int, default=None,
                       help='Encoder model focus value (same as "--model-focus"), default is None')
    group.add_argument('--loss-space', type=str, default='default',
                       help='Teacher model space, default is %(default)r, are names, split by commas')
    group.add_argument('--loss-scale', type=str, default='', metavar='k=v', nargs='*',
                       help='Scale value of loss, format is "loss_name=scale_value"')
    group.add_argument('--hard-data-selection', action='store_true', default=False,
                       help='Use hard data selection, default is %(default)s')
    group.add_argument('--no-data-selection', action='store_true', default=False,
                       help='Disable data selection, default is enabled')
    group.add_argument('--tau-scheduler', type=str, default='const',
                       help='Gumbel-Softmax temperature scheduler, default is %(default)r')
    group.add_argument('--gs-tau', type=float, default=1.0,
                       help='Tau value of gumbel softmax, default is %(default)s')
    group.add_argument('--gs-eps', type=float, default=1e-10,
                       help='Epsilon value of gumbel softmax, default is %(default)s')
    group.add_argument('--softmax-k', type=float, default=100.0,
                       help='Large softmax K value of "soft" dev accuracy, default is %(default)s')
    group.add_argument('--loss-obj', action='store_true', default=False,
                       help='Using dev loss as objective, instead of dev accuracy')
    group.add_argument('--eg-start', type=float, default=None,
                       help='The epsilon value of epsilon-greedy policy at the first batch, default is None (disabled)')
    group.add_argument('--eg-end', type=float, default=0.0,
                       help='The epsilon value of epsilon-greedy policy at the last batch, default is %(default)r')
    group.add_argument('--eg-end-step', type=int, default=200,
                       help='The end step of epsilon-greedy policy, '
                            'epsilon will be linear reduce to zero from start to here.'
                            'Default is %(default)r')

    return group


def add_train_args(parser):
    group = parser.add_argument_group('Training Options', description='Training options.')

    group.add_argument('--raw-train-epoch', type=int, default=0,
                       help='Number of raw training epochs, default is %(default)r')
    group.add_argument('-T', '--teacher-path', default=None, help='Preload teacher path, default is None')

    group.add_argument('--teacher-optimizer', default='sgd',
                       help='Teacher optimizer, default is %(default)r')
    group.add_argument('--teacher-lr', default=0.1, type=float,
                       help='Teacher initial learning rate')
    group.add_argument('--teacher-wdecay', default=1e-4, type=float,
                       help='Teacher weight decay, default is %(default)r')

    group.add_argument('--first-order', action='store_true', default=False,
                       help='Run first order optimization in teacher train step, default is False')

    group.add_argument('--teacher-bptt', type=int, default=70,
                       help='Sequence length of valid sample, default is %(default)r')

    return group


def set_default_args(args):
    args.tied = True

    assert args.max_student_epoch >= args.epochs

    if hasattr(args, 'model_space'):
        args.model_space = [int(w.strip()) for w in args.model_space.strip().split(',')]
    if hasattr(args, 'loss_space'):
        args.loss_space = args.loss_space.strip().split(',')
        assert len(args.loss_space) == 1 and args.loss_space[0] == 'default', 'Does not support loss space search now'
    if hasattr(args, 'enable_acc_emb'):
        assert not args.enable_acc_emb, 'Does not support accuracy embedding now'

    return args

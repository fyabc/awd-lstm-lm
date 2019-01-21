import argparse
from collections import namedtuple
import hashlib
import math
import os
import time

import numpy as np
import torch

import data
from libs.modules import model
from libs.modules.splitcross import SplitCrossEntropyLoss
from libs.modules.teacher import Teacher
from libs.modules.weight_drop import WeightDrop
from libs.utils import paths
from libs.utils import args as util_args
from libs.utils.optimizers import build_optimizer
from libs.utils.utils import batchify, get_batch, repackage_hidden, CycledBatchIterator, get_lr, set_lr
from libs import train_fns

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
# Add other options.
util_args.add_teacher_args(parser)
util_args.add_train_args(parser)

args = parser.parse_args()

util_args.set_default_args(args)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save({
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'teacher': teacher.state_dict(),
            'teacher_optimizer': teacher_optimizer.state_dict(),
        }, f)


def model_load(fn):
    global model, criterion, optimizer, teacher, teacher_optimizer
    with open(fn, 'rb') as f:
        state = torch.load(f)
    model = state['model']
    criterion = state['criterion']
    optimizer = state['optimizer']
    teacher = Teacher.build_teacher(args, model, state_dict=state['teacher'])
    teacher_optimizer = build_optimizer(args, args.teacher_optimizer, teacher.parameters(), prefix='teacher_')
    teacher_optimizer.load_state_dict(state['teacher_optimizer'])


fn = os.path.join(args.data, 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest()))
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

os.makedirs(os.path.dirname(args.save), exist_ok=True)

###############################################################################
# Build the model
###############################################################################

criterion = None

args.ntokens = ntokens = len(corpus.dictionary)
model = model.RNNModel.build_model(args)

###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    set_lr(optimizer, args.lr)
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


teacher = Teacher.build_teacher(args, model)
teacher_optimizer = build_optimizer(args, args.teacher_optimizer, teacher.parameters(), prefix='teacher_')


# Combine hparams and models.
Trainer = namedtuple('trainer', [
    'hparams', 'teacher', 'student', 'student_optimizer', 'teacher_optimizer', 'criterion'])


###############################################################################
# Training code
###############################################################################

def evaluate(trainer, data_source, epoch, batch_size=10, raw=True):
    model = trainer.student

    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        sample = {'data': data, 'targets': targets, 'hidden': hidden}

        if raw:
            teacher_out = trainer.teacher.teacher_selection_step(sample, epoch, train=False)
            encoder_kw = train_fns.prepare_encoder_kw(trainer.hparams, teacher_out)
        else:
            encoder_kw = train_fns.prepare_encoder_kw(trainer.hparams, None)

        output, hidden = model(data, hidden, **encoder_kw)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(raw=False):
    global valid_hidden

    # Turn on training mode which enables dropout.
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    total_teacher_loss = 0

    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = get_lr(optimizer)
        set_lr(optimizer, lr2 * seq_len / args.bptt)
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        sample = {'data': data, 'targets': targets, 'hidden': hidden}

        if raw:
            (output, hidden, rnn_hs, dropped_rnn_hs), raw_loss, loss = train_fns.student_forward(
                args, teacher, model,
                sample=sample, teacher_out=None,
                criterion=criterion,
            )

            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()
        else:
            # 1. Teacher selection step.
            teacher_out = teacher.teacher_selection_step(sample, epoch, train=True)

            # 2. Student train step.
            (output, hidden, rnn_hs, dropped_rnn_hs), raw_loss, loss = train_fns.student_forward(
                args, teacher, model,
                sample=sample, teacher_out=teacher_out,
                criterion=criterion,
            )
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            if args.clip:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
            optimizer.step()

            # 3. Teacher train step.
            # cycled_valid_itr.seq_len = seq_len  # [NOTE]: Can set this sequence length for next valid sample.
            valid_hidden = repackage_hidden(valid_hidden)
            valid_data, valid_targets = next(cycled_valid_itr)
            valid_sample = {'data': valid_data, 'targets': valid_targets, 'hidden': valid_hidden}

            teacher_train_out = train_fns.teacher_train_step(trainer, valid_sample, epoch, sample, train=True)

            total_teacher_loss += teacher_train_out['teacher_objective'].data

        # torch.cuda.empty_cache()

        total_loss += raw_loss.data
        set_lr(optimizer, lr2)
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            log_str = '| epoch {:3d}{}'.format(epoch, ' (raw)' if raw else '')
            log_str += ' | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | ' \
                       'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        batch, len(train_data) // args.bptt, get_lr(optimizer),
                        elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2))
            if not raw:
                log_str += ' | t-loss {:5.2f}'.format(total_teacher_loss / args.log_interval)
                total_teacher_loss = 0
            print(log_str)
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.
try:
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's
    # weight (i.e. Adaptive Softmax)
    optimizer = build_optimizer(args, args.optimizer, params)

    trainer = Trainer(hparams=args, teacher=teacher, student=model,
                      student_optimizer=optimizer, teacher_optimizer=teacher_optimizer, criterion=criterion)
    cycled_valid_itr = CycledBatchIterator(args, val_data)
    valid_hidden = model.init_hidden(eval_batch_size)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        raw = epoch <= args.raw_train_epoch
        train(raw=raw)

        model_save(paths.with_epoch(args.save, epoch))
        model_save(paths.with_epoch(args.save, 'last'))

        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(trainer, val_data, epoch, raw=raw)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(trainer, val_data, epoch, batch_size=eval_batch_size, raw=raw)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                set_lr(optimizer, get_lr(optimizer) / 10.)

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')


def final_evaluate(args, test_data, test_batch_size):
    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(trainer, test_data, epoch, batch_size=test_batch_size, raw=raw)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)


def main():
    final_evaluate(args, test_data, test_batch_size)


if __name__ == '__main__':
    main()

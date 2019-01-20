import torch
import torch.nn as nn

from .embed_regularize import embedded_dropout
from .locked_dropout import LockedDropout
from .weight_drop import WeightDrop


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1,
                 wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [
                torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                              1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l
                         in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid,
                                   hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid),
                                   save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in
                         range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            # if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.raw_outputs = []
        self.outputs = []

    def reset(self):
        if self.rnn_type == 'QRNN':
            [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def raw_forward(self, input, hidden, return_h=False, out_index=-100, skip_last=True):
        """

        Args:
            input:
            hidden:
            return_h:
            out_index:

        Returns:
            tuple
                result:
                hidden:

                If `return_h` is True, return 2 extra lists:

                raw_outputs:
                outputs:
        """
        self.raw_outputs.clear()
        self.outputs.clear()

        # Set final layer from nlayers and out_index.
        if 1 <= out_index <= self.nlayers - 1:
            final_layer = out_index
        else:
            final_layer = self.nlayers

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        # emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []

        # [NOTE]: Intermediate layers.
        # raw_output, hidden = self.rnn(emb, hidden)
        for l, rnn in enumerate(self.rnns[:-1]):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            self.raw_outputs.append(raw_output)
            raw_output = self.lockdrop(raw_output, self.dropouth)
            self.outputs.append(raw_output)

            if l == final_layer - 1:
                break

        raw_output, new_h = self._last_layer(raw_output, hidden[self.nlayers - 1], append=True)
        new_hidden.append(new_h)

        # Postprocessing.
        hidden = new_hidden

        output = raw_output
        result = output.view(output.size(0) * output.size(1), output.size(2))
        if return_h:
            return result, hidden, self.raw_outputs.copy(), self.outputs.copy()
        return result, hidden

    def _last_layer(self, last_input, last_hidden, append=True):
        # [NOTE]: The last layer, change the hidden size.
        raw_output, new_h = self.rnns[-1](last_input, last_hidden)
        if append:
            self.raw_outputs.append(raw_output)
        # self.hdrop(raw_output)
        raw_output = self.lockdrop(raw_output, self.dropout)
        if append:
            self.outputs.append(raw_output)
        return raw_output, new_h

    def forward(self, input, hidden, return_h=False, model_selection=None):
        if model_selection is None:
            return self.raw_forward(input, hidden, return_h=return_h)
        if isinstance(model_selection, int):
            return self.raw_forward(input, hidden, return_h=return_h, out_index=model_selection)

        result, hidden, *others = output = self.raw_forward(input, hidden, return_h=return_h)

        # # Combine outputs, except the last layer.
        # for m, o in zip(model_selection.transpose(0, 1), reversed(self.raw_outputs[:-1])):
        #     print('#', m.shape, o.shape)

        # TODO: Combine selection outputs
        return output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_(),
                     weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                         self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (
                self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    def loss_regularization(self, hparams, raw_loss, rnn_hs, dropped_rnn_hs):
        loss = raw_loss
        # Activation Regularization
        if hparams.alpha:
            loss = loss + sum(hparams.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if hparams.beta:
            loss = loss + sum(hparams.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        return loss

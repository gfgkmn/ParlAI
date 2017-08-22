# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import torch.utils.model_zoo as model_zoo

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------

model_urls = {
    'wmt-lstm':
    'https://s3.amazonaws.com/research.metamind.io/cove/wmtlstm-b142a7f2.pth'
}

model_cache = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '.torch')


class StackedBRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 dropout_rate=0,
                 dropout_output=False,
                 rnn_type=nn.LSTM,
                 concat_layers=False,
                 padding=False,
                 char_level=False,
                 birnn=True):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        self.char_level = char_level
        self.birnn = birnn
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(
                rnn_type(
                    input_size, hidden_size, num_layers=1,
                    bidirectional=birnn))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        # No padding necessary.
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        # Pad if we care or if its during eval.
        if self.padding or not self.training:
            return self._forward_padded(x, x_mask)
        # We don't care.
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Encode all layers
        outputs = [x]
        output_hiddens = []
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(
                    rnn_input, p=self.dropout_rate, training=self.training)
            # Forward
            rnn_output, rnn_last_hidden = self.rnns[i](rnn_input)
            outputs.append(rnn_output)
            output_hiddens.append(rnn_last_hidden)
            # rnn_last_hidden torch.size (2, 174, 128)
            # 2 bidirectional, 174 batch * len_w, 128 char_rnn hidden_size

        if isinstance(self.rnns[i], nn.LSTM):
            if self.birnn:
                output_hiddens = torch.cat(
                    [output_hiddens[-1][0][0], output_hiddens[-1][0][1]], 1)
            else:
                output_hiddens = output_hiddens[-1][0][0]
            # index fetch h0
        else:
            if self.birnn:
                output_hiddens = torch.cat(
                    [output_hiddens[-1][0], output_hiddens[-1][1]], 1)
            else:
                output_hiddens = output_hiddens[-1][0]
            # for gru or lstm, just one return value

            # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(
                output, p=self.dropout_rate, training=self.training)
        # todo if just return hidden state, is it necessary ?
        if self.char_level:
            return output_hiddens
        else:
            return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        # Compute sorted sequence lengths
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)

        if self.char_level:
            batch_size = x_mask.size(0)
            origin_idx_sort = idx_sort
            zeros_indexs = [i for i, v in enumerate(lengths) if v == 0]
            if zeros_indexs:
                first_zero_index = zeros_indexs[0]
            else:
                first_zero_index = len(origin_idx_sort)
            idx_sort = origin_idx_sort[:first_zero_index]
            lengths = lengths[:first_zero_index]

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        output_hiddens = []
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(
                    rnn_input.data,
                    p=self.dropout_rate,
                    training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(
                    dropout_input, rnn_input.batch_sizes)
            rnn_output, rnn_last_hidden = self.rnns[i](rnn_input)
            outputs.append(rnn_output)
            output_hiddens.append(rnn_last_hidden)

        if isinstance(self.rnns[i], nn.LSTM):
            if self.birnn:
                output_hiddens = torch.cat(
                    [output_hiddens[-1][0][0], output_hiddens[-1][0][1]], 1)
            else:
                output_hiddens = output_hiddens[-1][0][0]
            # index fetch h0
        else:
            if self.birnn:
                output_hiddens = torch.cat(
                    [output_hiddens[-1][0], output_hiddens[-1][1]], 1)
            else:
                output_hiddens = output_hiddens[-1][0]
            # for gru or lstm, just one return value

            # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        if self.char_level:
            pad_variable = torch.zeros(batch_size - first_zero_index,
                                       output_hiddens.size(-1))
            if output_hiddens.data.is_cuda:
                pad_variable = pad_variable.cuda()

            output_hiddens = Variable(
                torch.cat((output_hiddens.data, pad_variable)))
            output_hiddens = output_hiddens.index_select(0, idx_unsort)
            return output_hiddens
        else:
            # Transpose and unsort
            output = output.transpose(0, 1)
            output = output.index_select(0, idx_unsort)

            # Dropout on output layer
            if self.dropout_output and self.dropout_rate > 0:
                output = F.dropout(
                    output, p=self.dropout_rate, training=self.training)
            if x_mask.size(1) != output.size(1):
                # cause when use multi-gpu pytorch split a batch into multiple
                # batch into different core, but max_len is calculate through a
                # whole batch so maybe in a gpu core, actural max length is
                # much shorter than max_len you should concatenate zeros after
                # question_hiddens
                for_pad = output.unsqueeze(1)
                # batch * max_len * feature
                pad_hiddens = F.pad(
                    for_pad, (0, 0, 0,
                              x_mask.size(1) - output.size(1))).squeeze(1)
                output = pad_hiddens
            return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y

        # Compute scores
        scores = x_proj.bmm(y_proj.transpose(2, 1))

        # Mask padding
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))

        # Normalize with softmax
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))

        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            # In training we output log-softmax for NLL
            alpha = F.log_softmax(xWy)
        else:
            # ...Otherwise 0-1 probabilities
            alpha = F.softmax(xWy)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class GatedMatchRNN(nn.Module):
    """
    Gated Match RNN, Given sequence X and Y match sequence Y to each element in
    X
    * s[t][j] = V_t * tanh(W_uq * u_q[j] + W_up * u_p[t] + W_vp * h[t-1])
    * a[t][i] = exp(s[t][i]) / sum(exp(s[t]))
    * c[t] = sum(a[t][i] * u_q[i])
    *
    * g[t] = sigmoid(W_g * concat(u_p[t]; c[t]))
    * u_p[t], c[t] = g[t] * concat(u_p[t]; c[t]))

    * h[t] = rnn(h[t-1], concat(u_p[t]; c[t]))
    """

    def __init__(self,
                 input_size,
                 dropout_rate=0,
                 dropout_output=False,
                 rnn_cell_type=nn.LSTMCell,
                 padding=False,
                 is_bidirectional=False,
                 gated=True,
                 h_weight=True):
        # according to rnet papaer, gated-match-lstm hidden size must equal to
        # input size
        super(GatedMatchRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.hidden_state_size = input_size
        self.W_q = nn.Linear(input_size, input_size)
        self.W_up = nn.Linear(input_size, input_size)
        self.h_weight = h_weight
        if h_weight:
            self.W_vp = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, 1)
        self.W_g = nn.Linear(2 * input_size, 2 * input_size)
        self.gated = gated
        self.rnn_cell_type = rnn_cell_type
        self.rnn_cell = rnn_cell_type(2 * input_size, input_size)
        self.is_bidirectional = is_bidirectional

    def forward(self, x, x_mask, y, y_mask):
        if self.is_bidirectional:
            x_forward = self.uni_forward(x, x_mask, y, y_mask)
            x_backward_in = maskd_reverse(x, x_mask)
            x_backward = self.uni_forward(x_backward_in, x_mask, y, y_mask)
            x_backward = maskd_reverse(x_backward, x_mask)
            # x_backward = self.uni_forward(x, x_mask, y, y_mask)
            return torch.cat((x_forward, x_backward), 2)
        else:
            return self.uni_forward(x, x_mask, y, y_mask)

    def uni_forward(self, x, x_mask, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            x_mask = batch * len1
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        batch = x.size(0)

        h = Variable(torch.rand([batch, self.hidden_state_size]))
        if self.rnn_cell_type == nn.LSTMCell:
            c = Variable(torch.rand([batch, self.hidden_state_size]))
        # compute ct for match lstm cell state
        if x.data.is_cuda:
            h = h.cuda()
            if self.rnn_cell_type == nn.LSTMCell:
                c = c.cuda()

        time_steps = x.size(1)
        hiddens = []
        for t in range(time_steps):
            y_proj = self.W_q(y.view(-1, y.size(-1))).view(y.size())
            # batch * len2 * h
            x_proj = self.W_up(x[:, t, :]).unsqueeze(1).expand(y_proj.size())
            # batch * h -> batch * len2 * h
            if self.h_weight:
                hidden_proj = self.W_vp(h).unsqueeze(1).expand(y_proj.size())
                # 1 * h
                sum_batch = torch.tanh(x_proj + y_proj + hidden_proj)
            else:
                sum_batch = torch.tanh(x_proj + y_proj)
            s = self.V(sum_batch.view(-1, sum_batch.size(-1))).squeeze()
            s = s.view(sum_batch.size()[:-1])
            # batch * len2
            s.data.masked_fill_(y_mask.data, -float('inf'))
            alpha = F.softmax(s)
            # batch * len2
            ct = weighted_avg(y, alpha)
            # batch * h

            merge_input = torch.cat((x[:, t, :], ct), 1)
            # batch * 2h

            if self.gated:
                gt = F.sigmoid(self.W_g(merge_input))
                lstm_input = torch.mul(gt, torch.cat((x[:, t, :], ct), 1))
                lstm_input.masked_fill_(
                    x_mask[:, t].unsqueeze(1).expand(lstm_input.size()),
                    float(0))
            else:
                lstm_input = merge_input
                lstm_input.masked_fill_(
                    x_mask[:, t].unsqueeze(1).expand(lstm_input.size()),
                    float(0))

            if self.rnn_cell_type == nn.LSTMCell:
                h, c = self.rnn_cell(lstm_input, (h, c))
            else:
                h = self.rnn_cell(lstm_input, h)
            hiddens.append(h)

        output = torch.stack(hiddens, 1)

        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(
                output, p=self.dropout_rate, training=self.training)

        return output


class PointerNetwork(nn.Module):
    """
    pointer network, Given sequence X and Y return start positon and end
    position
    * s[t][j] = V_t * tanh(W_hp * Hp[j] + W_ha  ha[t-1])
    * a[t][i] = exp(s[t][i]) / sum(exp(s[t]))
    * p[t] = softmax(a[t])
    *
    * return argmax(p)
    *

    * ha[t] = rnn(ha[t-1], weighted_avg(Hp, a[t]))
    """

    def __init__(self,
                 input_size,
                 question_size,
                 rnn_cell_type=nn.LSTMCell,
                 question_init=True):
        super(PointerNetwork, self).__init__()
        self.hidden_state_size = input_size
        self.question_hidden_size = question_size
        self.W_hp = nn.Linear(input_size, input_size)
        self.W_ha = nn.Linear(input_size, input_size)
        self.W_y_tranform = nn.Linear(question_size, input_size)
        self.W_uq = nn.Linear(input_size, input_size)
        self.W_vq = nn.Linear(input_size, input_size)
        self.V = nn.Linear(input_size, 1)
        self.question_init = question_init
        self.rnn_cell_type = rnn_cell_type
        self.rnn_cell = rnn_cell_type(input_size, input_size)

    def forward(self, x, x_mask, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            x_mask = batch * len1
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            start_scores = alphas
            end_scores = betas
        """
        batch = x.size(0)
        VrQ = Variable(torch.randn(1, self.hidden_state_size))
        if x.data.is_cuda:
            VrQ = VrQ.cuda()

        if self.question_init:
            y_proj = self.W_y_tranform(y.view(-1, y.size(-1))).view(
                y.size()[:-1] + (self.hidden_state_size, ))
            question_transform = self.W_uq(y_proj.view(-1, y_proj.size(-1)))
            VrQ_expand = VrQ.repeat(batch, y_proj.size(1), 1)
            parameter_transform = self.W_vq(
                VrQ_expand.view(-1, y_proj.size(-1)))
            s = self.V(torch.tanh(question_transform + parameter_transform)
                       ).view(y_proj.size()[:-1])
            # batch * len2
            # s.masked_fill_(y_mask.data, -float('inf'))
            s.masked_fill_(y_mask, -float('inf'))
            # batch * len2 * h
            alpha = F.softmax(s)
            rq = weighted_avg(y_proj, alpha)
            h = rq
        else:
            h = Variable(torch.rand([batch, self.hidden_state_size]))

        if self.rnn_cell_type == nn.LSTMCell:
            c = Variable(torch.rand([batch, self.hidden_state_size]))
        # compute ct for match lstm cell state
        if x.data.is_cuda:
            h = h.cuda()
            if self.rnn_cell_type == nn.LSTMCell:
                c = c.cuda()

        scores = []
        for t in range(2):
            x_proj = self.W_hp(x.view(-1, x.size(-1)))
            # batch * len1, h
            hidden = h.unsqueeze(1).repeat(1, x.size(1), 1)
            hidden_proj = self.W_ha(hidden.view(-1, h.size(-1)))
            sum_batch = torch.tanh(x_proj + hidden_proj)
            s = self.V(sum_batch).view(x.size()[:-1])
            # batch * len1
            s.data.masked_fill_(x_mask.data, -float('inf'))
            alpha = F.softmax(s)
            # batch * len2
            score = F.log_softmax(s)
            if self.training:
                scores.append(score)
            else:
                scores.append(alpha)
            ct = weighted_avg(x, alpha)
            # batch * h

            # merge_input = torch.cat((x[:, t, :], ct), 1)
            merge_input = ct
            # batch * 2h

            if self.rnn_cell_type == nn.LSTMCell:
                h, c = self.rnn_cell(merge_input, (h, c))
            else:
                h = self.rnn_cell(merge_input, h)

        return scores[0], scores[1]


class FineGrainedGate(nn.Module):
    def __init__(self, word_size, feature_size):
        super(FineGrainedGate, self).__init__()
        self.word_size = word_size
        self.feature_size = feature_size
        self.trans = nn.Linear(self.feature_size + 1, 1)

    def forward(self, word_emb, char_emb, feature):
        """
        word_emb = batch * max_length * emb_dim
        char_emb = batch * max_length * emb_dim
        feature = batch * max_length * feature
        """
        batch = word_emb.size(0)
        length = word_emb.size(1)
        feature = feature.unsqueeze(2).repeat(1, 1, self.word_size, 1)
        # batch * max_length * 1 * feature
        word_emb = word_emb.unsqueeze(3)
        # batch * max_length * emb_dim * 1
        gate = torch.cat([feature, word_emb], 3).view(-1,
                                                      self.feature_size + 1)
        # batch * max_length * emb_dim * feature
        gate_ratio = F.sigmoid(self.trans(gate)).view(batch, length,
                                                      self.word_size)
        # batch * max_length * emb_dim
        word_emb = word_emb.squeeze(3)
        return word_emb * gate_ratio + char_emb * (1 - gate_ratio)


class MTLSTM(nn.Module):

    def __init__(self, n_vocab=None, vectors=None, residual_embeddings=False):
        super(MTLSTM, self).__init__()
        self.embed = False
        if n_vocab is not None:
            self.embed = True
            self.vectors = nn.Embedding(n_vocab, 300)
            if vectors is not None:
                self.vectors.weight.data = vectors
        self.rnn = nn.LSTM(300, 300, num_layers=2, bidirectional=True)
        self.rnn.load_state_dict(
            model_zoo.load_url(model_urls['wmt-lstm'], model_dir=model_cache))
        self.residual_embeddings = residual_embeddings

    def forward(self, inputs, lengths, hidden=None):
        if self.embed:
            inputs = self.vectors(inputs)
        lens, indices = torch.sort(lengths, 0, True)
        outputs, hidden_t = self.rnn(
            pack(inputs[indices], lens.data.tolist(), batch_first=True),
            hidden)
        outputs = unpack(outputs, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices]
        if self.residual_embeddings:
            outputs = torch.cat([inputs, outputs], 2)
        return outputs

# ------------------------------------------------------------------------------
# Functional
# ------------------------------------------------------------------------------


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = Variable(torch.ones(x.size(0), x.size(1)))
    if x.data.is_cuda:
        alpha = alpha.cuda()
    alpha = alpha * x_mask.eq(0).float()
    alpha = alpha / alpha.sum(1).expand(alpha.size())
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)


def maskd_reverse(x, x_mask):
    """
    x = batch * len * d
    x_mask = batch * len
    assert mask item is 1 if x item is meaningless
    """
    max_len = x.size(1)
    return_ten = Variable(torch.zeros(x.size()))
    if x.data.is_cuda:
        return_ten = return_ten.cuda()
    for i in range(x.size(0)):
        length = x_mask[i].data.eq(0).sum()
        idx = Variable(
            torch.LongTensor(
                list(range(length - 1, -1, -1)) +
                list(range(length, max_len))))
        if x.data.is_cuda:
            idx = idx.cuda()
        return_ten[i] = x[i].index_select(0, idx)
    return x

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

# No modification is made to this file.
# Origin: https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa

# ------------------------------------------------------------------------------
# Modules
# ------------------------------------------------------------------------------


class HighwayLayer(nn.Module):
    def __init__(self, size, bias_init=0.0, nonlin=nn.ReLU(inplace=True), gate_nonlin=F.sigmoid):
        super(HighwayLayer, self).__init__()

        self.nonlin = nonlin
        self.gate_nonlin = gate_nonlin
        self.lin = nn.Linear(size, size)
        self.gate_lin = nn.Linear(size, size)
        self.gate_lin.bias.data.fill_(bias_init)

    def forward(self, x):
        out = self.nonlin(self.lin(x))
        gate_out = self.gate_nonlin(self.gate_lin(x))
        prod = torch.mul(out, gate_out)
        resid = torch.mul((1-gate_out), x)
        return torch.add(prod, resid)


class HighwayNet(nn.Module):
    def __init__(self, size, depth):
        super(HighwayNet, self).__init__()
        layers = [HighwayLayer(size) for _ in range(depth)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Conv1dMax(nn.Module):
    def __init__(self, in_chan, out_chan, width, do_p=0.5):
        self.do = nn.Dropout(do_p)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=[1, width])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(self.do(x)))
        _, out = torch.max(out, 2)
        return out


class Conv1dN(nn.Module):
    def __init__(self, nchan, filter_sizes, filter_heights, do_p):
        super(Conv1dN, self).__init__()

        conv_layers = [Conv1dMax(nchan, size, height, do_p)
                       for size, height in zip(filter_size, filter_heights)]
        self.main = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.main(x)


class LinearBase(nn.Module):
    def __init__(self, input_size, output_size, do_p=0.2):
        super(LinearBase, self).__init__()
        self.do = nn.Dropout(do_p)
        self.lin = nn.Linear(input_size, output_size)
        self.input_size = input_size

    def forward(self, a, b, mask):
        shape = a.size()
        N = self.input_size
        M = a.numel() // size
        a_ = a.view(M, N)
        b_ = b.view(M, N)
        return shape, a_, b_


class Linear(LinearBase):
    def forward(self, a, b, mask):
        shape, a_, b_ = super(self).forward(a, b, mask)
        input = torch.cat((a_, b__), 1)
        out = self.lin(self.do(input))
        out = out.view(shape).squeeze(len(shape)-1)
        return exp_mask(out, mask)


class TriLinear(LinearBase):
    def forward(self, a, b, mask):
        shape, a_, b_ = super(self).forward(a, b, mask)
        input = torch.cat((a_, b_, a_*b_), 1)
        out = self.lin(self.do(input))
        out = out.view(shape).squeeze(len(shape)-1)
        return exp_mask(out, mask)


class TFLinear(nn.Module):
    def __init__(self, input_size, output_size, func, do_p=0.2):
        super(TFLinear, self).__init__()
        if func == 'linear':
            self.main = Linear(input_size, output_size, do_p)
        elif func == 'trilinear':
            self.main = TriLinear(input_size, output_size, do_p)
        else:
            assert False

    def forward(self, a, b, mask):
        return self.main(a, b, mask)
    
            
class BiEncoder(nn.Module):
    def __init__(self, config, input_size):
        super(Encoder, self).__init__()
        self.config = config
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                           num_layers=config.n_layers, dropout=config.dp_ratio,
                           bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, _ = self.rnn(inputs, (h0, c0))
        return outputs


class FixedEmbedding(nn.Embedding):
    def forward(input):
        out = super(FixedEmbedding, self).forward(input)
        return Variable(out.data)


class BiAttention(nn.Module):
    def __init__(self, args, logits_size):
        super(BiAttention, self).__init__()
        self.lin = TFLinear(size, args.attn_func)
        self.args = args

    def forward(self, text, query, text_mask, query_mask):
        a = self.args
        max_sent_size, max_num_sents, max_q_size = \
            a.max_sent_size, a.max_num_sents, a.max_q_size
        text_aug = text.unsqueeze(3).repeat(1, 1, 1, max_q_size, 1)
        query_aug = query.unqueeze(1).unsqueeze(1).repeat(1, max_num_sents, max_sent_size, 1, 1)
        text_mask_aug = text_mask.unsqueeze(3).repeat(1, 1, 1, max_q_size)
        query_mask_aug = query_mask.unqueeze(1).unsqueeze(1).repeat(1, max_num_sents, max_sent_size, 1)
        text_query_mask = text_mask_aug * query_mask_aug
        query_logits = self.lin(text_aug, query_aug, text_query_mask)

        _, query_logits_max = torch.max(query_logits, 3)
        # c2q
        text_attn = softsel(text, query_logits_max).unsqueeze(2).repeat(1, 1, max_sent_size, 1)
        # q2c
        query_attn = softsel(query_aug, query_logits)

        attn = torch.cat((text, query_attn, text * query_attn, text * text_attn), 3)
        return attn


class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      bidirectional=True))

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
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(0, 1)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
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

        # Sort x
        x = x.index_select(0, idx_sort)

        # Transpose batch and sequence dims
        x = x.transpose(0, 1)

        # Pack it up
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # Encode all layers
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to input
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data,
                                          p=self.dropout_rate,
                                          training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                                                        rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])

        # Unpack everything
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]

        # Concat hidden layers or take final
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]

        # Transpose and unsort
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
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

def softsel(target, logits):
    out = F.softmax(logits)
    out = out.unsqueeze(len(out.size())).mul(target).sum(len(target.size())-2)
    return out


def exp_mask(logits, mask):
    return torch.add_(logits, (1 - mask)) * VERY_NEGATIVE_NUMBER


def softmax3d(input, xd, yd):
    out = input.view(-1, xd*yd)
    out = F.softmax(out).view(-1, xd, yd)
    return out


def reduce_max(input_tensor, axis):
    _, values = input_tensor.max(axis)
    return values


def span_loss(config, q_mask, logits_start, start, logits_end, end):
    size = config.max_num_sents * config.max_sent_size
    loss_mask = reduce_mask(q_mask, 1)
    losses_start = nn.CrossEntropyLoss(logits_start, start.view(-1, size))
    ce_loss_start = torch.mean(loss_mask * losses)
    losses_end = nn.CrossEntropyLoss(logits_end, end.view(-1, size))
    ce_loss_end = torch.mean(loss_mean)
    return ce_loss_end - ce_loss_start
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers
from .utils import charvob_size
from torch.autograd import Variable
import torch.nn.functional as F


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of RNET."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    RNN_CELL_TYPES = {
        'lstm': nn.LSTMCell,
        'gru': nn.GRUCell,
        'rnn': nn.RNNCell
    }

    def __init__(self, opt, padding_idx=0):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(opt['vocab_size'],
                                      opt['embedding_dim'],
                                      padding_idx=padding_idx)

        self.char_embedding = nn.Embedding(charvob_size,
                                           opt['char_embedding_dim'],
                                           padding_idx=padding_idx)

        # ...(maybe) keep them fixed
        if opt['fix_embeddings']:
            for p in self.embedding.parameters():
                p.requires_grad = False

        # Register a buffer to (maybe) fill later for keeping *some* fixed
        if opt['tune_partial'] > 0:
            buffer_size = torch.Size((
                opt['vocab_size'] - opt['tune_partial'] - 2,
                opt['embedding_dim']
            ))
            self.register_buffer('fixed_embedding', torch.Tensor(buffer_size))

        # RNN docuemnt character encoder
        self.char_rnn = layers.StackedBRNN(
            input_size=opt['char_embedding_dim'],
            hidden_size=opt['charemb_rnn_dim'],
            num_layers=opt['doc_char_layers'],
            dropout_rate=opt['dropout_char_rnn'],
            dropout_output=opt['dropout_char_rnn_output'],
            concat_layers=False,
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            # padding=opt['rnn_padding'],
            padding=True,
            char_level=True
        )

        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'] +
                                                  opt['charemb_rnn_dim'] * 2)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features'] + \
            opt['charemb_rnn_dim'] * 2
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim'] + opt['charemb_rnn_dim'] * 2

        # RNN document encoder
        # disabled use gated_match_lstm and self-alignment layer to subsitude
        self.doc_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        question_input_size = opt['embedding_dim'] + opt['charemb_rnn_dim'] * 2

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=question_input_size,
            hidden_size=opt['hidden_size'],
            num_layers=opt['question_layers'],
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
        )

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        self.gated_match_rnn = layers.GatedMatchRNN(
            input_size=doc_hidden_size,
            # assert doc_rnn and question_rnn's hidden_state not concat
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            rnn_cell_type=self.RNN_CELL_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            is_bidirectional=True, )

        self.self_alignment_rnn = layers.GatedMatchRNN(
            input_size=doc_hidden_size * 2,
            # assert doc_rnn and question_rnn's hidden_state not concat
            dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'],
            rnn_cell_type=self.RNN_CELL_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            gated=False,
            is_bidirectional=False,
            h_weight=False)

        self.pointer_network = layers.PointerNetwork(
            input_size=doc_hidden_size * 2,
            question_size=question_hidden_size,
            rnn_cell_type=self.RNN_CELL_TYPES[opt['rnn_type']], )

        # # Question merging
        # if opt['question_merge'] not in ['avg', 'self_attn']:
        #     raise NotImplementedError(
        #         'question_merge = %s' % opt['question_merge'])
        # if opt['question_merge'] == 'self_attn':
        #     self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # # Bilinear attention for span start/end
        # self.start_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     question_hidden_size,
        # )
        # self.end_attn = layers.BilinearSeqAttn(
        #     doc_hidden_size,
        #     question_hidden_size,
        # )

    def forward(self, x1, x1_f, x1_mask, x1_chars, x1_chars_mask, x2, x2_mask,
                x2_chars, x2_chars_mask, redoc, reques):
        """Inputs:
        x1 = document word indices                 [ batch * len_d]
        x1_f = document word features indices      [ batch * len_d * nfeat]
        x1_mask = document padding mask            [ batch * len_d]
        x1_chars = document character indices      [ batch2 * len_d * len_c]
        x1_chars_mask = document character indices [ batch2 * len_d * len_c]
        x2 = question word indices                 [ batch * len_q]
        x2_mask = question padding mask            [ batch * len_q]
        x2_chars = document character indices      [ batch3 * len_q * len_c]
        x2_chars_mask = document character indices [ batch3 * len_q * len_c]
        redoc = rebuild document char encoding     [ batch * len_d]
        reques = rebuild question char encoding    [ batch * len_q]
        """

        if len(x1_f.size()) == 1:
            # x1_f size : (batch,)
            no_manual_feature = True
        else:
            no_manual_feature = False

        batch_size = x1.size(0)
        doc_length = x1.size(1)
        question_length = x2.size(1)

        x1_chars_emb = self.char_embedding(x1_chars)
        x2_chars_emb = self.char_embedding(x2_chars)

        # Dropout on character-level embeddings
        if self.opt['dropout_char_emb'] > 0:
            x1_chars_emb = nn.functional.dropout(
                x1_chars_emb,
                p=self.opt['dropout_char_emb'],
                training=self.training)
            x2_chars_emb = nn.functional.dropout(
                x2_chars_emb,
                p=self.opt['dropout_char_emb'],
                training=self.training)

        # character-level encoding
        doc_char_encoding = self.char_rnn(x1_chars_emb, x1_chars_mask)
        question_char_encoding = self.char_rnn(x2_chars_emb, x2_chars_mask)

        # rebuild document and question char-encoding
        doc_char_rebuild = Variable(
            torch.Tensor(batch_size * doc_length, doc_char_encoding.size(-1))
            .fill_(0))
        question_char_rebuild = Variable(
            torch.Tensor(batch_size * question_length,
                         doc_char_encoding.size(-1)).fill_(0))

        if x1.data.is_cuda:
            doc_char_rebuild = doc_char_rebuild.cuda()
            question_char_rebuild = question_char_rebuild.cuda()

        # rebuild doc char encoding
        redoc = redoc.view(-1)
        # batch * len_d
        redoc_order, idx_sort = torch.sort(redoc, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        doc_batch = doc_char_encoding.index_select(
            0, redoc_order[redoc_order.ne(-1)])
        doc_batch_pad = doc_batch.unsqueeze(0).unsqueeze(0)
        doc_batch_pad = F.pad(
            doc_batch_pad,
            (0, 0, 0, batch_size * doc_length - doc_batch.size(0))).squeeze()
        doc_char_rebuild = doc_batch_pad.index_select(0, idx_unsort)
        doc_char_rebuild = doc_char_rebuild.view(batch_size, doc_length, -1)

        # rebuild ques char encoding
        reques = reques.view(-1)
        # batch * len_d
        reques_order, idx_sort = torch.sort(reques, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        ques_batch = question_char_encoding.index_select(
            0, reques_order[reques_order.ne(-1)])
        ques_batch_pad = ques_batch.unsqueeze(0).unsqueeze(0)
        ques_batch_pad = F.pad(
            ques_batch_pad,
            (0, 0, 0,
             batch_size * question_length - ques_batch.size(0))).squeeze()
        question_char_rebuild = ques_batch_pad.index_select(0, idx_unsort)
        question_char_rebuild = question_char_rebuild.view(
            batch_size, question_length, -1)

        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'],
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'],
                                           training=self.training)

        # concatenate word-level and character-level encoding
        x1_emb = torch.cat([x1_emb, doc_char_rebuild], 2)
        x2_emb = torch.cat([x2_emb, question_char_rebuild], 2)

        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            # drnn_input = torch.cat([x1_emb, x2_weighted_emb, x1_f], 2)
            drnn_input = torch.cat([x1_emb, x2_weighted_emb], 2)
        else:
            drnn_input = x1_emb

        if not no_manual_feature:
            drnn_input = torch.cat([drnn_input, x1_f], 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)

        doc_hiddens = self.gated_match_rnn(doc_hiddens, x1_mask,
                                           question_hiddens, x2_mask)
        doc_hiddens = self.self_alignment_rnn(doc_hiddens, x1_mask,
                                              doc_hiddens, x1_mask)
        # print(doc_hiddens.size())
        scores = self.pointer_network(doc_hiddens, x1_mask, question_hiddens,
                                      x2_mask)
        start_scores, end_scores = scores

        # # Encode question with RNN + merge hiddens
        # if self.opt['question_merge'] == 'avg':
        #     q_merge_weights = layers.uniform_weights(
        #             question_hiddens, x2_mask)
        # elif self.opt['question_merge'] == 'self_attn':
        #     q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        # question_hidden = layers.weighted_avg(question_hiddens,
        #                                       q_merge_weights)

        # # Predict start and end positions
        # start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        # end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores

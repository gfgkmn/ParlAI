# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers
from .utils import charvob_size


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of RNET."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

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
        # todo may padding is needed, it's different with doc_rnn
        self.doc_char_rnn = layers.StackedBRNN(
            input_size=opt['char_embedding_dim'],
            hidden_size=opt['charemb_rnn_dim'],
            num_layers=opt['doc_char_layers'],
            dropout_rate=opt['dropout_char_rnn'],
            dropout_output=opt['dropout_char_rnn_output'],
            concat_layers=False,
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            return_hidden=True
        )

        self.question_char_rnn = layers.StackedBRNN(
            input_size=opt['char_embedding_dim'],
            hidden_size=opt['charemb_rnn_dim'],
            num_layers=opt['question_char_layers'],
            dropout_rate=opt['dropout_char_rnn'],
            dropout_output=opt['dropout_char_rnn_output'],
            concat_layers=False,
            rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'],
            return_hidden=True
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

        # Question merging
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError(
                'question_merge = %s' % opt['question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
        )

    def forward(self, x1, x1_f, x1_mask, x1_chars, x1_chars_mask, x2, x2_mask,
                x2_chars, x2_chars_mask):
        """Inputs:
        x1 = document word indices                 [ batch * len_d]
        x1_f = document word features indices      [ batch * len_d * nfeat]
        x1_mask = document padding mask            [ batch * len_d]
        x1_chars = document character indices      [ batch * len_d * len_c]
        x1_chars_mask = document character indices [ batch * len_d * len_c]
        x2 = question word indices                 [ batch * len_q]
        x2_mask = question padding mask            [ batch * len_q]
        x2_chars = document character indices      [ batch * len_q * len_c]
        x2_chars_mask = document character indices [ batch * len_q * len_c]
        """

        x1_chars_size = x1_chars.size()
        x2_chars_size = x2_chars.size()

        # todo current batch must == 1, otherwise the long sequence last word
        # will be all zero.
        x1_chars_emb = self.char_embedding(
            x1_chars.view(-1, x1_chars.size(-1)))
        x2_chars_emb = self.char_embedding(
            x2_chars.view(-1, x2_chars.size(-1)))
        # emb shape [batch * len_d , len_c, char_emb_dim]

        # todo cache mechanism to cache same word charater-level encoding for
        # computer just once in batch

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
        doc_char_encoding = self.doc_char_rnn(x1_chars_emb, x1_chars_mask)
        question_char_encoding = self.question_char_rnn(
            x2_chars_emb, x2_chars_mask)

        doc_char_encoding = doc_char_encoding.view(
            x1_chars_size[:-1] + doc_char_encoding.size()[-1:])
        question_char_encoding = question_char_encoding.view(
            x2_chars_size[:-1] + question_char_encoding.size()[-1:])

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
        x1_emb = torch.cat([x1_emb, doc_char_encoding], 2)
        x2_emb = torch.cat([x2_emb, question_char_encoding], 2)

        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input = torch.cat([x1_emb, x2_weighted_emb, x1_f], 2)
        else:
            drnn_input = torch.cat([x1_emb, x1_f], 2)

        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
        # todo maybe this is debug place
        # print(doc_hiddens.size())

        # Encode question with RNN + merge hiddens
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens,
                                              q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores

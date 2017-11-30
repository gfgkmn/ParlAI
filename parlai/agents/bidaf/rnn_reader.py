# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
import torch.nn as nn
from . import layers

# Modification: add 'pos' and 'ner' features.
# Origin:
# https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents/drqa


# class BIDAF(nn.Module):
class RnnDocReader(nn.Module):
    def __init__(self, args, dic=None):
        super(BIDAF, self).__init__()

        batch_size, d_hidden = args.batch_size, args.d_hidden
        max_num_sents, max_sent_size = args.max_num_sents, args.max_sent_size
        max_ques_size, max_word_size = args.max_ques_size, args.max_word_size
        word_vocab_size, char_vocab_size = args.word_vocab_size, \
            args.char_vocab_size
        d_char_embed, d_embed = args.d_char_embed, args.d_embed
        d_char_out = args.d_char_out
        self.dict_misc = dic_misc

        seq_in_size = 4 * d_hidden
        lin_config = [seq_in_size] * 2
        self.char_embed = layers.FixedEmbedding(char_vocab_size, d_char_embed)
        self.word_embed = layers.FixedEmbedding(word_vocab_size, d_embed)
        self.h_net = layers.HighwayNet(d_embed, args.n_hway_layers)
        # self.pre_encoder = layers.BiEncoder(word_embed_size, args)
        # self.attend = layers.BiAttention(size, args)
        # self.start_encoder0 = layers.BiEncoder(word_embed_size, args)
        # self.start_encoder1 = layers.BiEncoder(word_embed_size, args)
        # self.end_encoder = layers.BiEncoder(word_embed_size, args)
        self.lin_start = layers.TFLinear(*lin_config, args.answer_func)
        self.lin_end = layers.TFLinear(*lin_config, args.answer_func)

        self.enc_start_shape = (batch_size, max_num_sents * max_sent_size,
                                d_hidden * 2)
        self.logits_reshape = (batch_size, max_num_sents * max_sent_size)
        self.args = args

    # def forward(self, ctext, text, text_mask, cquery, query, query_mask):
    def forward(self, text, text_mask, ctext, query, query_mask, cquery):
        a = self.args

        # Character Embedding Layer
        ctext_embed = self.char_embed(ctext)
        cquery_embed = self.char_embed(cquery)
        ctext_embed = self.conv(ctext_embed)
        cquery_embed = self.conv(cquery_embed)

        # Word Embedding Layer
        text_embed = self.word_embed(text)
        query_embed = self.word_embed(query)

        # a learned joint character / word embedding
        text = self.h_net(torch.cat((ctext_embed, text_embed), 3))
        query = self.h_net(torch.cat((cquery_embed, query_embed), 2))

        # Contextual Embedding Layer
        text = self.pre_encoder(text)
        query = self.pre_encoder(query)

        # Attention Flow Layer
        text_attn = self.attend(text, query, text_mask, query_mask)

        # The input to the modeling layer is G, which encodes the
        # query-aware representations of context words.
        # Modeling Layer
        text_attn_enc_start = self.start_encoder0(text_attn)
        text_attn_enc_start = self.start_encoder1(text_attn_enc_start)

        # p1 = softmax(w^T_{p1}[G;M])
        # Output Layer
        logits_start = self.lin_start(text_attn_enc_start, text_attn,
                                      text_mask)
        start = layers.softmax3d(logits_start, a.max_num_sents,
                                 a.max_sent_size)

        # softmax of weights from start - not really explained in the paper
        a1i = layers.softsel(
                text_attn_enc_start.view(self.enc_start_shape),
                logits_start.view(self.logits_reshape)
                ).unsqueeze(1).unsqueeze(1).repeat(
                        1, a.max_num_sents, a.max_sent_size, 1)

        span = torch.cat(
            (text_attn, text_attn_enc_start, a1i, text_attn_enc_start * a1i),
            3)
        text_attn_enc_end = self.end_encoder(span)
        logits_end = self.lin_end(text_attn_enc_end, text_attn, text_mask)
        end = layers.softmax3d(logits_end, a.max_num_sents, a.max_sent_size)
        return start, end


class drqa(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    # def __init__(self, opt, padding_idx=0, embedding=None):
    def __init__(self, opt, dic=None, padding_idx=0):
        super(RnnDocReader, self).__init__()
        # Store config
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            # assert embedding is not None
            self.embedding = nn.Embedding(
                opt['vocab_size'],
                opt['embedding_dim'],
                padding_idx=padding_idx)
            # self.embedding.weight.data = embedding
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < opt['vocab_size']
                # fixed_embedding = embedding[opt['tune_partial'] + 2:]
                buffer_size = torch.Size(
                    (opt['vocab_size'] - opt['tune_partial'] - 2,
                     opt['embedding_dim']))
                self.register_buffer('fixed_embedding',
                                     torch.Tensor(buffer_size))
                # self.fixed_embedding = fixed_embedding
        else:  # random initialized
            self.embedding = nn.Embedding(
                opt['vocab_size'],
                opt['embedding_dim'],
                padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
        # Projection for attention weighted question
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

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

        # RNN question encoder
        self.question_rnn = layers.StackedBRNN(
            input_size=opt['embedding_dim'],
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

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(
                x1_emb, p=self.opt['dropout_emb'], training=self.training)
            x2_emb = nn.functional.dropout(
                x2_emb, p=self.opt['dropout_emb'], training=self.training)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            drnn_input_list.append(x1_ner_emb)
        drnn_input = torch.cat(drnn_input_list, 2)
        # Encode document with RNN
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)

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

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
import torch
# import time
import unicodedata
from collections import Counter
import spacy

NLP = spacy.load('en_core_web_sm')

# pos_list = [
#     'DET', 'ADP', 'PART', 'ADJ', 'PUNCT', 'INTJ', 'NOUN', 'ADV', 'X', 'PRON',
#     'PROPN', 'VERB', 'CONJ', 'SPACE', 'NUM', 'SYM', 'CCONJ'
# ]
# ner_list = [
#     'QUANTITY', 'PRODUCT', 'EVENT', 'FACILITY', 'NORP', 'TIME', 'LANGUAGE',
#     'ORG', 'DATE', 'CARDINAL', 'PERSON', 'ORDINAL', 'LOC', 'PERCENT',
#     'MONEY', 'WORK_OF_ART', 'GPE', 'FAC', 'LAW', ''
# ]
# ------------------------------------------------------------------------------
# Data/model utilities.
# ------------------------------------------------------------------------------


def normalize_text(text):
    return unicodedata.normalize('NFD', text)
    # what whid NFC, NFD, NFKC, NFKD mean? normalize ?


def load_embeddings(opt, word_dict):
    """Initialize embeddings from file of pretrained vectors."""
    embeddings = torch.Tensor(len(word_dict), opt['embedding_dim'])
    embeddings.normal_(0, 1)

    # Fill in embeddings
    if not opt.get('embedding_file'):
        raise RuntimeError('Tried to load embeddings with no embedding file.')
    with open(opt['embedding_file']) as f:
        for line in f:
            parsed = line.rstrip().split(' ')
            assert(len(parsed) == opt['embedding_dim'] + 1)
            w = normalize_text(parsed[0])
            if w in word_dict:
                vec = torch.Tensor([float(i) for i in parsed[1:]])
                embeddings[word_dict[w]].copy_(vec)

    # Zero NULL token
    embeddings[word_dict['__NULL__']].fill_(0)

    return embeddings


def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    # add manual features to this agent.
    # i know why you initialize DocumentReader with featuredict
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
        feature_dict['in_question_lemma'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    # if opt['use_ner']:
    #     for ner_type in ner_list:
    #         feature_dict['ner=%s' % ner_type] = len(feature_dict)
    # if opt['use_pos']:
    #     for pos_type in pos_list:
    #         feature_dict['pos=%s' % pos_type] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------


def vectorize(opt, ex, dict_misc, feature_dict):
    """Turn tokenized text inputs into feature vectors."""
    # Index words
    # ex['document'], ex['question']
    assert type(ex['document']) == spacy.tokens.doc.Doc
    assert type(ex['question']) == spacy.tokens.doc.Doc
    document = torch.LongTensor([dict_misc[w.text] for w in ex['document']])
    question = torch.LongTensor([dict_misc[w.text] for w in ex['question']])

    # Create extra features vector
    features = torch.zeros(len(ex['document']), len(feature_dict))
    # try:
    #     poss = torch.LongTensor(
    #         [dict_misc.pos2ind[w.pos_] for w in ex['document']])
    #     ners = torch.LongTensor(
    #         [dict_misc.ner2ind[w.ent_type_] for w in ex['document']])
    # except KeyError:
    #     import ipdb
    #     ipdb.set_trace()
    # feature matrix, len(docuemnt) * len(feature) shape

    # f_{exact_match}
    if opt['use_in_question']:
        q_words_cased = set([w.text for w in ex['question']])
        q_words_uncased = set([w.text.lower() for w in ex['question']])
        q_words_lemma = set([
            w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()
            for w in ex['question']
        ])
        for i in range(len(ex['document'])):
            w = ex['document'][i]
            if w.text in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if w.text.lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0
            if (w.lemma_ if w.lemma_ != '-PRON-' else
                    w.text.lower()) in q_words_lemma:
                features[i][feature_dict['in_question_lemma']] = 1.0

    # f_{tf}
    if opt['use_tf']:
        counter = Counter([d.text.lower() for d in ex['document']])
        ll = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[
                w.text.lower()] * 1.0 / ll
    if opt['use_time'] > 0:
        # Counting from the end, each (full-stop terminated) sentence gets
        # its own time identitfier.
        sent_idx = 0

        def _full_stop(w):
            return w.text in {'.', '?', '!'}
        for i, w in reversed(list(enumerate(ex['document']))):
            sent_idx = sent_idx + 1 if _full_stop(w) else max(sent_idx, 1)
            if sent_idx < opt['use_time']:
                features[i][feature_dict['time=T%d' % sent_idx]] = 1.0
            else:
                features[i][feature_dict['time>=T%d' % opt['use_time']]] = 1.0
    # just like sentence reverse identifier. when use-time=8. and there are 16
    # sentence.  so first 8 sentence feature is time>t8 and then time=t7
    # time=t6 etc. and alway is a feature matrix

    char_docs = torch.LongTensor(
        [[dict_misc.char2ind[c] for c in w.text] for w in ex['document']])
    char_questions = torch.LongTensor(
        [[dict_misc.char2ind[c] for c in w.text] for w in ex['question']])

    # Maybe return without target
    if ex['target'] is None:
        # return document, features, question, poss, ners
        return char_docs, document, char_questions, question

    # ...or with target

    # so in actually we didn't provide labels in squad DefaultTeacher.
    # this condition didn't satisfy
    start = torch.LongTensor(1).fill_(ex['target'][0])
    end = torch.LongTensor(1).fill_(ex['target'][1])

    # return document, features, question, poss, ners, start, end
    return char_docs, document, char_questions, question, start, end


def batchify(batch, null=0, cuda=False):
    """
    Collate inputs into batches.
    generate input matrix and vector for batch.
    """
    # NUM_INPUTS = 5
    NUM_INPUTS = 4
    NUM_TARGETS = 2
    NUM_EXTRA = 4

    # Get elements
    # provide by vectorize function
    # docs = [ex[0] for ex in batch]
    # features = [ex[1] for ex in batch]
    # questions = [ex[2] for ex in batch]
    # poss = [ex[3] for ex in batch]
    # ners = [ex[4] for ex in batch]
    docs = [ex[0] for ex in batch]
    char_docs = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    char_questions = [ex[3] for ex in batch]

    # provide by bidaf.py _build_ex function
    token_ques = [ex[-3] for ex in batch]
    token_doc = [ex[-4] for ex in batch]
    text = [ex[-2] for ex in batch]
    spans = [ex[-1] for ex in batch]
    # we couldn't sure ex[5] and ex[6] is exist.

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    # max_length is not global setting, it's a batch setting.
    x1 = torch.LongTensor(len(docs), max_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    # x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
    # x1_pos = torch.LongTensor(len(docs), max_length).fill_(null)
    # x1_ner = torch.LongTensor(len(docs), max_length).fill_(null)
    # (samples, doc_lengths(time_steps), features)
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        # x1_f[i, :d.size(0)].copy_(features[i])
        # x1_pos[i, :d.size(0)].copy_(poss[i])
        # x1_ner[i, :d.size(0)].copy_(ners[i])
    # fill document matrix.

    # Batch char docuemnt
    max_char_length = max([max([len(w) for w in d]) for d in char_docs])
    x1_chars = torch.LongTensor(len(docs), max_length,
                                max_char_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x1_chars_mask = torch.ByteTensor(len(docs), max_length,
                                     max_char_length).fill_(1)
    # (samples, doc_lengths(time_steps), features)
    for i, d in enumerate(char_docs):
        for j, c in enumerate(d):
            x1_chars[i, j, :c.size(0)].copy_(c)
            x1_chars_mask[i, j, :c.size(0)].fill_(0)
    # fill document_chars matrix.

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).fill_(null)
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
    # fill question matrix.

    # Batch char docuemnt
    max_char_length = max(
        [max([len(w) for w in d]) for d in char_questions])
    x2_chars = torch.LongTensor(len(questions), max_length,
                                max_char_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x2_chars_mask = torch.ByteTensor(len(questions), max_length,
                                     max_char_length).fill_(1)
    # (samples, doc_lengths(time_steps), features)
    for i, d in enumerate(char_questions):
        for j, c in enumerate(d):
            x2_chars[i, j, :c.size(0)].copy_(c)
            x2_chars_mask[i, j, :c.size(0)].fill_(0)
    # fill question_chars matrix.

    # Pin memory if cuda
    if cuda:
        x1 = x1.pin_memory()
        # looks-like some memory optimize technicle
        # x1_f = x1_f.pin_memory()
        x1_mask = x1_mask.pin_memory()
        x1_chars = x1_chars.pin_memory()
        x1_chars_mask = x1_chars_mask.pin_memory()
        # x1_pos = x1_pos.pin_memory()
        # x1_ner = x1_ner.pin_memory()
        x2 = x2.pin_memory()
        x2_mask = x2_mask.pin_memory()
        x2_chars = x2_chars.pin_memory()
        x2_chars_mask.x2_chars_mask.pin_memory()

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        # return x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, token_doc, \
        #         token_ques, text, spans
        return x1, x1_mask, x1_chars, x2, x2_mask, x2_chars, \
                token_doc, token_ques, text, spans

    # ...Otherwise add targets
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        y_s = torch.cat([ex[NUM_INPUTS] for ex in batch])
        y_e = torch.cat([ex[NUM_INPUTS + 1] for ex in batch])
        # return x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, y_s, y_e, \
        #     token_doc, token_ques, text, spans
        return x1, x1_mask, x1_chars, x2, x2_mask, x2_chars, \
            y_s, y_e, token_doc, token_ques, text, spans
    # start-position and end position vector

    # ...Otherwise wrong number of inputs
    raise RuntimeError('Wrong number of inputs per batch')


# ------------------------------------------------------------------------------
# General logging utilities.
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

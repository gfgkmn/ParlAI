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

NLP = spacy.load('en')
pos_list = [
    'DET', 'ADP', 'PART', 'ADJ', 'PUNCT', 'INTJ', 'NOUN', 'ADV', 'X', 'PRON',
    'PROPN', 'VERB', 'CONJ', 'SPACE', 'NUM', 'SYM', 'CCONJ'
]
ner_list = [
    'QUANTITY', 'PRODUCT', 'EVENT', 'FACILITY', 'NORP', 'TIME', 'LANGUAGE',
    'ORG', 'DATE', 'CARDINAL', 'PERSON', 'ORDINAL', 'LOC', 'PERCENT', 'MONEY',
    'WORK_OF_ART', 'GPE', 'FAC', 'LAW'
]
pos_dict = {i: pos_list.index(i)/len(pos_list) for i in pos_list}
ner_dict = {i: ner_list.index(i)/len(ner_list) for i in ner_list}

# charset = string.ascii_letters + string.digits + string.punctuation
charset = set([0, 10, 8211, 257] + list(range(32, 241))) - set(
    [127, 192, 193, 211, 221, 222, 223, 238])
charset = list(charset)
char_dict = {i: charset.index(i) for i in charset}
char_dict[0] = len(char_dict)
charvob_size = len(charset) + 1


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
    embeddings[word_dict['<NULL>']].fill_(0)

    return embeddings


def build_feature_dict(opt):
    """Make mapping of feature option to feature index."""
    # add manual features to this agent.
    # i know why you initialize DocumentReader with featuredict
    feature_dict = {}
    if opt['use_in_question']:
        feature_dict['in_question'] = len(feature_dict)
        feature_dict['in_question_uncased'] = len(feature_dict)
    if opt['use_tf']:
        feature_dict['tf'] = len(feature_dict)
    if opt['use_ner']:
        feature_dict['ner_type'] = len(feature_dict)
    if opt['use_pos']:
        feature_dict['pos_type'] = len(feature_dict)
    if opt['use_time'] > 0:
        for i in range(opt['use_time'] - 1):
            feature_dict['time=T%d' % (i + 1)] = len(feature_dict)
        feature_dict['time>=T%d' % opt['use_time']] = len(feature_dict)
    return feature_dict


# ------------------------------------------------------------------------------
# Torchified input utilities.
# ------------------------------------------------------------------------------


def vectorize(opt, ex, word_dict, feature_dict):
    """Turn tokenized text inputs into feature vectors."""
    # Index words
    document = torch.LongTensor([word_dict[w] for w in ex['document']])
    question = torch.LongTensor([word_dict[w] for w in ex['question']])
    # cause there is no charater whose ord value equal 0, so use 0 represent
    # unknow, or out of character table.
    redoc_p2w = ex['document']
    usedoc_w2p = {w: p for p, w in enumerate(set(ex['document']))}
    rebuild_doc_info = (redoc_p2w, usedoc_w2p)
    document_chars = [
        torch.LongTensor([
            char_dict[ord(i)] if ord(i) in char_dict else char_dict[0]
            for i in w
        ]) for w in usedoc_w2p
    ]
    reques_p2w = ex['question']
    useques_w2p = {w: p for p, w in enumerate(set(ex['question']))}
    rebuild_ques_info = (reques_p2w, useques_w2p)
    question_chars = [
        torch.LongTensor([
            char_dict[ord(i)] if ord(i) in char_dict else char_dict[0]
            for i in w
        ]) for w in useques_w2p
    ]

    # Create extra features vector
    features = torch.zeros(len(ex['document']), len(feature_dict))
    # feature matrix, len(docuemnt) * len(feature) shape

    spacy_doc = NLP(' '.join(ex['document']))

    # f_{exact_match}
    if opt['use_in_question']:
        q_words_cased = set([w for w in ex['question']])
        q_words_uncased = set([w.lower() for w in ex['question']])
        for i in range(len(ex['document'])):
            if ex['document'][i] in q_words_cased:
                features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in q_words_uncased:
                features[i][feature_dict['in_question_uncased']] = 1.0

    # f_{tf}
    if opt['use_tf']:
        counter = Counter([w.lower() for w in ex['document']])
        ll = len(ex['document'])
        for i, w in enumerate(ex['document']):
            features[i][feature_dict['tf']] = counter[w.lower()] * 1.0 / ll
    if opt['use_ner']:
        for i, w in enumerate(ex['document']):
            if spacy_doc[i].ent_type_:
                features[i][feature_dict['ner_type']] = ner_dict[spacy_doc[
                    i].ent_type_]

    if opt['use_pos']:
        for i, w in enumerate(ex['document']):
            if spacy_doc[i].pos_:
                features[i][feature_dict['pos_type']] = pos_dict[spacy_doc[
                    i].pos_]

    if opt['use_time'] > 0:
        # Counting from the end, each (full-stop terminated) sentence gets
        # its own time identitfier.
        sent_idx = 0

        def _full_stop(w):
            return w in {'.', '?', '!'}
        for i, w in reversed(list(enumerate(ex['document']))):
            sent_idx = sent_idx + 1 if _full_stop(w) else max(sent_idx, 1)
            if sent_idx < opt['use_time']:
                features[i][feature_dict['time=T%d' % sent_idx]] = 1.0
            else:
                features[i][feature_dict['time>=T%d' % opt['use_time']]] = 1.0
    # just like sentence reverse identifier. when use-time=8. and there are 16
    # sentence.  so first 8 sentence feature is time>t8 and then time=t7
    # time=t6 etc. and alway is a feature matrix

    # Maybe return without target
    if ex['target'] is None:
        return document, document_chars, rebuild_doc_info, features, \
                question, question_chars, rebuild_ques_info

    # ...or with target
    start = torch.LongTensor(1).fill_(ex['target'][0])
    end = torch.LongTensor(1).fill_(ex['target'][1])

    return document, document_chars, rebuild_doc_info, features, question, \
        question_chars, rebuild_ques_info, start, end


def batchify(batch, null=0, cuda=False):
    """
    Collate inputs into batches.
    generate input matrix and vector for batch.
    """
    NUM_INPUTS = 7
    NUM_TARGETS = 2
    NUM_EXTRA = 2

    # Get elements
    docs = [ex[0] for ex in batch]
    doc_chars = [ex[1] for ex in batch]
    rebuild_doc_infos = [ex[2] for ex in batch]
    features = [ex[3] for ex in batch]
    questions = [ex[4] for ex in batch]
    question_chars = [ex[5] for ex in batch]
    rebuild_ques_infos = [ex[6] for ex in batch]
    text = [ex[-2] for ex in batch]
    spans = [ex[-1] for ex in batch]
    # we couldn't sure ex[7] and ex[8] is exist.

    # Batch documents and features
    max_length = max([d.size(0) for d in docs])
    # max_length is not global setting, it's a batch setting.
    x1 = torch.LongTensor(len(docs), max_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x1_mask = torch.ByteTensor(len(docs), max_length).fill_(1)
    if len(features[0].size()) > 1:
        x1_f = torch.zeros(len(docs), max_length, features[0].size(1))
        no_manual_feature = False
    else:
        x1_f = torch.zeros(len(docs))
        no_manual_feature = True
    # (samples, doc_lengths(time_steps), features)
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if not no_manual_feature:
            x1_f[i, :d.size(0)].copy_(features[i])
    # fill document matrix.

    # Batch char document
    batch_w2t = dict()
    max_char_length = 0
    for bi, index_info in enumerate(rebuild_doc_infos):
        # to get word -> character tensor corresponding
        # and max word length
        p2w, w2p = index_info
        for w in w2p:
            if w not in batch_w2t:
                batch_w2t[w] = doc_chars[bi][w2p[w]]
                if len(doc_chars[bi][w2p[w]]) > max_char_length:
                    max_char_length = len(doc_chars[bi][w2p[w]])
    batch_order_chars = {w: new_pos for new_pos, w in enumerate(batch_w2t)}

    x1_chars = torch.LongTensor(len(batch_w2t), max_char_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x1_chars_mask = torch.ByteTensor(len(batch_w2t), max_char_length).fill_(1)
    # (samples, doc_lengths(time_steps), features)
    for w in batch_order_chars:
        x1_chars[batch_order_chars[w], :batch_w2t[w].size(0)].copy_(
            batch_w2t[w])
        # word in batch new position, word correspond character encoding
        x1_chars_mask[batch_order_chars[w], :batch_w2t[w].size(0)].fill_(0)
    doc_char_rebuild = torch.LongTensor(len(docs), max_length).fill_(-1)
    for bi, index_info in enumerate(rebuild_doc_infos):
        p2w, w2p = index_info
        doc_char_rebuild[bi, :len(p2w)].copy_(
            torch.LongTensor([batch_order_chars[w] for w in p2w]))
    # fill document_chars matrix.

    # Batch questions
    max_length = max([q.size(0) for q in questions])
    x2 = torch.LongTensor(len(questions), max_length).fill_(null)
    x2_mask = torch.ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
    # fill question matrix.

    # Batch char question
    batch_w2t = dict()
    max_char_length = 0
    for bi, index_info in enumerate(rebuild_ques_infos):
        # to get word -> character tensor corresponding
        # and max word length
        p2w, w2p = index_info
        for w in w2p:
            if w not in batch_w2t:
                batch_w2t[w] = question_chars[bi][w2p[w]]
                if len(question_chars[bi][w2p[w]]) > max_char_length:
                    max_char_length = len(question_chars[bi][w2p[w]])
    batch_order_chars = {w: new_pos for new_pos, w in enumerate(batch_w2t)}

    x2_chars = torch.LongTensor(len(batch_w2t), max_char_length).fill_(null)
    # (samples, doc_lengths(time_steps))
    x2_chars_mask = torch.ByteTensor(len(batch_w2t), max_char_length).fill_(1)
    # (samples, doc_lengths(time_steps), features)
    for w in batch_order_chars:
        x2_chars[batch_order_chars[w], :batch_w2t[w].size(0)].copy_(
            batch_w2t[w])
        # word in batch new position, word correspond character encoding
        x2_chars_mask[batch_order_chars[w], :batch_w2t[w].size(0)].fill_(0)
    question_char_rebuild = torch.LongTensor(len(questions),
                                             max_length).fill_(-1)
    for bi, index_info in enumerate(rebuild_ques_infos):
        p2w, w2p = index_info
        question_char_rebuild[bi, :len(p2w)].copy_(
            torch.LongTensor([batch_order_chars[w] for w in p2w]))
    # fill document_chars matrix.

    # Pin memory if cuda
    if cuda:
        x1 = x1.pin_memory()
        # looks-like some memory optimize technicle
        x1_f = x1_f.pin_memory()
        x1_mask = x1_mask.pin_memory()
        x1_chars = x1_chars.pin_memory()
        x1_chars_mask = x1_chars_mask.pin_memory()
        x2 = x2.pin_memory()
        x2_mask = x2_mask.pin_memory()
        x2_chars = x2_chars.pin_memory()
        x2_chars_mask = x2_chars_mask.pin_memory()
        doc_char_rebuild = doc_char_rebuild.pin_memory()
        question_char_rebuild = question_char_rebuild.pin_memory()

    # Maybe return without targets
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA:
        return x1, x1_f, x1_mask, x1_chars, x1_chars_mask, x2, x2_mask, \
               x2_chars, x2_chars_mask, doc_char_rebuild, \
               question_char_rebuild, text, spans

    # ...Otherwise add targets
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        y_s = torch.cat([ex[7] for ex in batch])
        y_e = torch.cat([ex[8] for ex in batch])
        return x1, x1_f, x1_mask, x1_chars, x1_chars_mask, x2, x2_mask, \
            x2_chars, x2_chars_mask, doc_char_rebuild, question_char_rebuild, \
            y_s, y_e, text, spans
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

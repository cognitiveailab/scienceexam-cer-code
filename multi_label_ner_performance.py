from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import pickle
import os
import numpy as np

def pickle_load_large_file(filepath):
    max_bytes = 2**31 - 1
    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)
    return obj
def get_entities(seq, suffix=False):
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + [['O']]]
    #print('seq: ',seq[0:100])
    prev_tag = ['O']
    prev_type = ['']
    previous_begin_offset = {}
    chunks = []
    for i, chunk in enumerate(seq + [['O']]):
        if chunk == []:
            chunk = ['O']
        tag = [singlechunk[0] for singlechunk in chunk]
        type_ = [singlechunk.split('-')[-1] for singlechunk in chunk]
        for idx, prev_tag_temp in enumerate(prev_tag):
            prev_type_temp = prev_type[idx]
            if end_of_chunk(prev_tag_temp, tag, prev_type_temp, type_):
                chunks.append((prev_type_temp, begin_offset[prev_type_temp], i - 1))
        begin_offset = {}
        for idx, singlechunk in enumerate(chunk):
            tag_temp = tag[idx]
            type__temp = type_[idx]
            if start_of_chunk(tag_temp, type__temp):
                begin_offset[type__temp] = i
            else:
                if type__temp not in previous_begin_offset.keys():
                    begin_offset[type__temp] = i
                else:
                    begin_offset[type__temp]=previous_begin_offset[type__temp]
        previous_begin_offset=begin_offset.copy()
        prev_tag = tag
        prev_type = type_
    return chunks

def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False


    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and 'I' not in tag: chunk_end = True
    if prev_tag == 'B' and 'I' in tag:
        for idx, tag_temp in enumerate(tag):
            if tag_temp == 'I':
                if type_[idx] == prev_type:
                    chunk_end = False
                    break
            chunk_end = True
    if prev_tag == 'I' and 'I' not in tag: chunk_end = True
    if prev_tag == 'I' and 'I' in tag:
        for idx, tag_temp in enumerate(tag):
            if tag_temp == 'I':
                if type_[idx] == prev_type:
                    chunk_end = False
                    break
            chunk_end = True
    return chunk_end

def start_of_chunk(tag, type_):

    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    return chunk_start

def f1_score(y_true, y_pred, average='micro', suffix=False):

    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t==y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def classification_report(y_true, y_pred, digits=2, suffix=False):
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))
    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))
    last_line_heading = 'avg / total'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # compute averages
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),
                             np.average(rs, weights=s),
                             np.average(f1s, weights=s),
                             np.sum(s),
                             width=width, digits=digits)

    return report
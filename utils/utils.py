#!/usr/bin/env python

# -*- encoding: utf-8

'''
_____.___._______________  __.____ __________    _________   ___ ___    _____  .___ 
\__  |   |\_   _____/    |/ _|    |   \      \   \_   ___ \ /   |   \  /  _  \ |   |
 /   |   | |    __)_|      < |    |   /   |   \  /    \  \//    ~    \/  /_\  \|   |
 \____   | |        \    |  \|    |  /    |    \ \     \___\    Y    /    |    \   |
 / ______|/_______  /____|__ \______/\____|__  /  \______  /\___|_  /\____|__  /___|
 \/               \/        \/               \/          \/       \/         \/     
 

@author: Yekun Chai
@license: CASIA
@email: chaiyekun@gmail.com
@file: utils.py.py
@time: 5/5/20 7:46 PM
@desc：       
               
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

assert tf.__version__.startswith('2')
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from typing import List

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import codecs
import numpy as np
import time
import json
import pickle
from datetime import datetime
from functools import partial
import logging
import random
from absl import app
from absl import flags

from evaluator.BLEU.bleu import BLEUEvaluator
from evaluator.CIDEr.cider import CIDErEvaluator
from evaluator.METEOR.meteor import METEOREvaluator
from evaluator.ROUGE.rouge import RougeEvaluator

evaluators = [BLEUEvaluator(4), METEOREvaluator(), RougeEvaluator(), CIDErEvaluator()]

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

RANDOM_SEED = 2020
PAD = '[PAD]'
UNK = '[UNK]'
START = '[START]'
END = '[END]'

pad_id, unk_id, start_id, end_id = range(4)

assert tf.executing_eagerly() is True


def reset_seed(seed=RANDOM_SEED):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def print_configuration_op(FLAGS):
    print('My Configurations:')
    # pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        print(f'\x1b[1;36;m {name}:\t{value.value} \x1b[0m')
    # for k, v in sorted(FLAGS.__dict__.items()):
    # print(f'{k}={v}\n')
    print(f'End of configuration\n {"=" * 80}')


def evaluate(hypos: dict, refs: dict, idx_word, scorer, save_path=None):
    assert hypos.keys() == refs.keys(), 'Not all hypothesis provided!'
    hypos_sentences = {k: [tokens_to_sentence(v, idx_word)] for k, v in hypos.items()}
    refs_sentences = {k: seqs_to_sentences(v, idx_word) for k, v in refs.items()}
    # hypos_sentences = seqs_to_sentences(hypos, idx_word)
    # refs_sentences = seqs_to_sentences(refs, idx_word)
    # hypos_sentences = {k: [v] for k, v in enumerate(hypos_sentences)}
    # refs_sentences = {k: [v] for k, v in enumerate(refs_sentences)}
    results = scorer(hypos_sentences, refs_sentences)
    if save_path:
        with open(save_path, 'w') as f:
            for i in hypos_sentences:
                hypo = hypos_sentences[i]
                ref = refs_sentences[i]
                line = f'{hypo}\t{ref}\n'
                f.write(line)
    return results


def measure_score(hypos, refs, mode='all'):
    scores = []
    for evaluator in evaluators:
        scores.extend(evaluator.compute_score(refs, hypos, mode))
    score_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'METEOR', 'ROUGE', 'CIDEr']
    scores_dict = dict(zip(score_names, scores))
    return scores_dict


def every_measure_score(hypos, refs):
    cider_eval = evaluators[-1]
    cider_scores = cider_eval.compute_score(refs, hypos, 'every')
    scores = np.squeeze(np.array(cider_scores))
    return scores


def quick_measure_score(hypos, refs):
    meteor_score = evaluators[1].compute_score(refs, hypos)[0]
    cider_score = evaluators[3].compute_score(refs, hypos)[0]
    return {'METEOR': meteor_score, 'CIDEr': cider_score}


def seqs_to_sentences(tokens_list, idx_word):
    seq_list = []
    for tokens in tokens_list:
        sent = tokens_to_sentence(tokens, idx_word)
        seq_list.append(sent)
    return seq_list


def tokens_to_sentence(tokens, idx_word):
    ws = []
    for i, idx in enumerate(tokens):
        if i == 0 and idx == START:
            continue
        if idx == pad_id or idx == end_id:
            break
        w = idx_word[idx]
        if w is None:
            break
        ws.append(w)
    if len(ws) == 0 or ws[-1] != '.':
        ws.append('.')
    return " ".join(ws)


def save_log(file, log, mode='w'):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, mode) as f:
        f.write(str(log) + '\n')

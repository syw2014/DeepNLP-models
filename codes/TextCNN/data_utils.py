#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : data_utils.py
# PythonVersion: python3.6
# Date    : 2019/10/30 10:47
# Software: PyCharm
"""Tools to create tf.data.dataset for model train/eval/inference"""

import tensorflow as tf
import pandas as pd
import numpy as np

def padding(ids, max_seq_length, before=False):
    """
    Padding id sequence to the maximum sequence.Note here you can choose padding at the head or end of the sequence.
    And here we use 0 as the padding element
    :param ids: input int32 list
    :param max_seq_length: maximum sequence length
    :param before: whether to pad [0] at the head or end of the sequence, default was fasle
    :return: padded sequence list
    """
    length = len(ids)
    if length < max_seq_length:
        if before:
            ids = [0] * (max_seq_length - length) + ids
        else:
            ids.extend([0] * (max_seq_length - length))
        return ids
    else:
        return ids[: max_seq_length]


def create_dataset_with_tf(filename, vocab, epochs, batch_size, max_seq_length, mode):
    """
    Create dataset with tf.data.Dataset, pandas---> tf.data.Dataset
    Here we use tf.data.Dataset.from_tensor_slices to create tf.dataset, so firstly it will read all data into memory
    and may occupy large memory if your corpus was large.
    :param filename: input data file name, file format as csv and separator default was tab, here only two fileds
    :param vocab: Vocab object create in `create_vocab.py`
    :param epochs: how many epochs will run
    :param batch_size: batch size
    :param max_seq_length: maximum sequence length
    :param mode: which model will run in train/eval/inference
    :return:
    """
    def get_label_id(row):
        """Convert label text to label id."""
        return vocab.labels[row.strip()]

    header_names = ["cate", "text"]
    data = pd.read_csv(filename, sep='\t', header=None, names=header_names, encoding='utf-8')
    data["label"] = data['cate'].apply(get_label_id)

    data = data.dropna()
    # sequence segment and convert token to ids
    data["split"] = data['text'].apply(lambda line: vocab.doc_to_ids(line))

    # sequence padding
    data['inputs'] = data['split'].apply(lambda tokens: padding(tokens, max_seq_length, before=True))

    num_samples = data.shape[0]

    # tf.data.dataset
    dataset = tf.data.Dataset.from_tensor_slices((data['inputs'], data['label']))

    if mode == "train":
        dataset = dataset.repeat(epochs)
        dataset.shuffle(num_samples)
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    return dataset, num_samples


def create_single_input(sequence, vocab, max_seq_length):
    """
    Convert a sequence to input format
    :param sequence: input text
    :param vocab: Vocab object define in `create_vocab.py`
    :param max_seq_length: maximum sequence length
    :return:
    """
    # tokenize and token to ids
    token_ids = vocab.doc_to_ids(sequence)
    padded_ids = padding(token_ids, max_seq_length)

    return padded_ids

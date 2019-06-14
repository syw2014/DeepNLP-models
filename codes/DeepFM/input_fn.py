#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : input_fn.py
# PythonVersion: python3.6
# Date    : 2019/6/5 上午10:44
# IDE     : PyCharm

"""Create input data pipeline with tf.dataset api."""
import tensorflow as tf
from util_tools import Params


def _parse_line(line, params):
    """Parse line with csv format
    :param line: input line
    :param params: instance of Params(), parameter dictionary
    :return values, value index, label
    """
    # columns = ["value", "index", "label"]
    field_default = [[""], [""], [0.0]]
    fields = tf.decode_csv(line, field_default, field_delim="\t")
    print(fields)

    # split strings
    val_str, ids_str, label = fields[0], fields[1], fields[-1]

    values = tf.string_split([val_str], delimiter=",").values
    indices = tf.string_split([ids_str], delimiter=",").values

    # Convert string values to tf.float32 or tf.int32
    print("type: ", values)
    values = tf.string_to_number(values, tf.float32)
    indices = tf.string_to_number(indices, tf.int32)

    # padding value and indices sequence
    # TODO, use hyper-parameters to complete padding
    padding = tf.constant([[0, 0], [0, params.field_size]], dtype=tf.int32)
    values_padded = tf.pad([values], padding, constant_values=0.0)
    values = tf.slice(values_padded, [0, 0], [-1, params.field_size])
    values = tf.squeeze(values, [0])

    # pad ids
    ids_padded = tf.pad([indices], padding, constant_values=params.padding_value)
    indices = tf.slice(ids_padded, [0, 0], [-1, params.field_size])
    indices = tf.squeeze(indices, [0])
    return values, indices, label


def load_dataset_from_text(filename, params):
    """
    Create tf.data instance from input file
    :param filename: input file name
    :param params: Instant of Parmas object
    :return: dataset(tf.dataset)
    """
    dataset = tf.data.TextLineDataset(filename)

    dataset = dataset.map(lambda line: _parse_line(line, params))

    return dataset


def input_fn(dataset, params=None, mode='train'):
    """
    Input function for deepFM model
    :param dataset: tf.dataset, yielding list of tuple(values, indices, label)
    :param params: Instance of Params object
    :return: dict with inputs
    """
    is_training = (mode == "train")
    buffer_size = params.train_size if is_training else 1
    # create batches
    input_dataset = (dataset
                     .shuffle(buffer_size=buffer_size)
                     .batch(params.batch_size)
                     .prefetch(1))  # always keep one batch ready to serve

    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = input_dataset.make_initializable_iterator()
    (values, indices, labels) = iterator.get_next()
    init_op = iterator.initializer
    print('label shape:', tf.reshape(labels, shape=[-1, 1]))

    inputs = {
        "values": values,
        "indices": indices,
        "labels": tf.reshape(labels, shape=[-1, 1]),
        "iterator_init_op": init_op
    }

    return inputs


if __name__ == "__main__":
    dirs = "/root/data/research/data/product/rs/kaggle_DAC/processed/"
    param_file = dirs + "dataset_params.json"
    params = Params(param_file)

    filename = dirs + "tmp.tsv"
    dataset = load_dataset_from_text(filename, params)
    inputs = input_fn(dataset, params, "train")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(inputs['iterator_init_op'])

        print("input values shape: ", inputs['values'].get_shape)
        print("input indices shape: ", inputs['indices'].get_shape)
        print("input y shape: ", inputs['labels'].get_shape)

        print("input values: ", sess.run(inputs['values']))
        print("input indices: ", sess.run(inputs['indices']))
        print("input y: ", sess.run(inputs['labels']))
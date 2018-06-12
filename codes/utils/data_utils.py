#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: data_utils.py
# Date: 18-6-11 上午9:46

"""An useful tools to prepare dataset for deep learning model, which can generate shuffled batches."""

import tensorflow as tf
from preprocess import *


def expand(x):
    """Expand the scalar to a vector of length 1.
    Args:
        x: the parsed example
    Return:
        expanded x
    """
    x["pair_id"] = tf.expand_dims(tf.convert_to_tensor(x["pair_id"]), 0)
    x["label"] = tf.expand_dims(tf.convert_to_tensor(x["label"]), 0)

    return x


def deflate(x):
    x["pair_id"] = tf.squeeze(x["pair_id"])
    x["label"] = tf.squeeze(x["label"])

    return x


def prepare_dataset(record_file, batch_size=128):
    """Make a tfrecord to a shuffled , batched dataset for tensorflow input.
    Args:
        record_file: tfrecord filenames
        batch_size: batch size
    Return:
          A shuffled, batched dataset
    """
    dataset = tf.data.TFRecordDataset([record_file])

    # Apply map the parse function to each record
    dataset = dataset.map(BuildTFRecord.example_parse, num_parallel_calls=5)
    # shuffled
    dataset = dataset.shuffle(buffer_size=10000)

    # expand
    # dataset = dataset.map(expand)
    dataset = dataset.padded_batch(batch_size, padded_shapes={
        "pair_id": [],       # scalar do not need padding
        "label": [],         # scala do not need padding
        "q_tokens": tf.TensorShape([None]),
        "d_tokens": tf.TensorShape([None])
    })

    # dataset = dataset.map(deflate)

    return dataset


def prepare_dataset_iterators(record,batch_size=128):
    train_dataset = prepare_dataset(record, batch_size=batch_size)

    # define an abstract object
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_dataset)

    return next_element, training_init_op


if __name__ == "__main__":
    tfrecord = "../../data/train.tfrecord"

    with tf.Session() as sess:
        next_element, train_init_op = prepare_dataset_iterators(tfrecord, batch_size=1)

        sess.run(train_init_op)

        # print(next_element)
        # print(sess.run(next_element["d_tokens"]))
        for _ in range(10):
            print(sess.run(next_element["q_tokens"]))
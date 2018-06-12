#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Date: 2018-06-08 09:58:30
# Auther: Jerry.Shi


"""A simple implementaiton to create tfrecord for model train."""
#from __future__ import absolute_print
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import os 
import numpy as np
from datetime import datetime
import sys
import threading
import random

from preprocess import *

tf.flags.DEFINE_string("dataset_dir", "../../data/atec_nlp_sim_train.txt", "input dataset")
tf.flags.DEFINE_string("output_dir", "../../data/tfrecords/", "output file dir")
tf.flags.DEFINE_string("train_shards", 1, "number of shards in training TFRecord files")
tf.flags.DEFINE_string("num_threads", 4, "number of threads")



def _bytes_features(value):
    """Wrapper for insterting a bytes or string freature to Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _bytes_feature_list(values):
    """Wrapper for instering bytes FeatureList into SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_features(v) for v in values])

def _text_to_sequence_example(data):
    """Builds a SequenceExample proto for your input data.For text data
    Args:
        data: The data you want to pred, a list of element.
    Returns:
        A SequenceExample proto.
    """

    # Here tfrecord only for text, so there no context
    feature_list = tf.train.FeatureLists(feature_list={
        "word_ids": _bytes_feature_list(data)})
    
    sequence_example = tf.train.SequenceExample(
            # here context is noe, you can define your context
            feature_lists=feature_list)

    return sequence_example


class BuildTFRecord(Vocab):
    """Build tfrecod."""
    def __int__(self):
        self._create_vocab()


    def sequence_to_example(self,query_id_list, doc_id_list, pair_id, label):
        """Convert a pair of text id list to tf example.
        Args:
            query_id_list: The token id list of query
            doc_id_list: The token id list  of doc
            pair_id: The id of this pair in dataset
            label: A integer of 0 or 1 to indicate query and doc are similar
        Returns:
            tf example
        """
        example = tf.train.SequenceExample()

        example.context.feature["pair_id"].int64_list.value.append(pair_id)
        example.context.feature["label"].int64_list.value.append(label)

        # Feature lists
        q_tokens = example.feature_lists.feature_list["q_tokens"]
        d_tokens = example.feature_lists.feature_list["d_tokens"]

        for token in query_id_list:
            q_tokens.feature.add().int64_list.value.append(token)
        for token in doc_id_list:
            d_tokens.feature.add().int64_list.value.append(token)

        return example


def _process_one_file(thread_index, ranges, name, meta_data, num_shards):
    """Process and save a subset of meta data as TFRecord files in one thread.
    Args:
        thread_index: Integer thread identifier with in [0, len(ranges)]
        ranges: A list of pairs of integers specifying the ranges of the dataset to process in parallel
        name: Unqiue identifier specifying the dataset
        meta_data: List of raw data
        num_shards: Integer, number of shards for output files
    """
    # Each thread process N shards wher N = num_shards / num_threads. For instance, if num_threads = 2
    # and num_shards = 128, then the first thread would produce shards [0, 64)
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    start_point = ranges[thread_index][0]
    end_point = ranges[thread_index][1]
    shard_ranges = np.linspace(start_point, end_point, num_shards_per_batch+1).astype(int)

    num_data_in_thread = end_point - start_point

    counter = 0
    for s in range(num_shards_per_batch):
        # generate a shards version of the file name, eg: 'train-00001-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join("", output_filename)

        # write to file
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        data_in_shard = np.range(shard_ranges[s], shard_ranges[s+1], dtype=int)
        for i in data_in_shard:
            data = meta_data[i]

            # Serialize, here you can choose different serialize mehtod
            sequence_example = _text_to_sequence_example(data)
            #sequence_example = sequence_to_example()
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            # print info
            if not counter % 100 :
                print("%s [thread %d]: Processed %d of %d items in thread batch " % 
                        (datetime.now(), thread_index, shard_counter, num_data_in_thread))

            sys.stdout.flush()
        
        writer.close()
    print("{} [shards {}]: Wrote {} data to {} shards".format(datetime.now(),
        thread_index, counter, num_data_in_thread))
    sys.stdout.flush()


def _process_dataset(name, dataset, num_shards, set_threads):
    """Process a complete datset and save as TFRecod
    Args:
        name: Unique identifier specifying the dataset
        dataset: List of input data
        num_shards: Integer number of shards for output files
        set_threads: Integer number of thread you want to use
    """
    # Shuffle the ordering of images, make the randomization repeatable
    random.seed(1234)
    random.shuffle(dataset)
    # TODO, more process of your data
    num_threads = min(num_shards, set_threads)
    # assign data in each thread
    spacing = np.linspace(0, len(dataset), num_threads+1).astype(int)

    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i+1]])

    # create a mechansim for monitoring when all thread are finished
    coord = tf.train.Coordinator()

    # Launch a thread for each batch
    print("Launch {} threads for spacing: {}".format(num_shards, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, dataset, num_shards)
        t = threading.Thread(target=_process_one_file, args=args)
        t.start()
        threads.append(t)

    # Wait for all threads to terminate
    coord.join(threads)
    print("{}: Finished processing all {} data pairs in dataset {}".format(
        datetime.now(), len(dataset), name))





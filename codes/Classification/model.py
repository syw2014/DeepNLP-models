#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model.py
# PythonVersion: python3.5
# Date    : 2019/10/19 10:02
# Software: PyCharm
"""Create models with tf.keras.Models(Tensorflow2.0)"""

import tensorflow as tf


class TextCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_seq_length, num_filters, filter_size,
                 num_classes, dropout_rate, hidden_size, is_training=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        self.num_filters = list(map(int, num_filters.split(',')))
        self.filter_size = list(map(int, filter_size.split(',')))
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.is_training = is_training
        self.num_classes = num_classes

        # define layers used build networks, we can use tf.keras.layers or define our own layers
        # here we define text cnn with conv1d
        self.embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size,
                                                   output_dim=self.embedding_dim,
                                                   input_length=self.max_seq_length)
        self.conv1 = tf.keras.layers.Conv1D(self.num_filters[0],
                                            self.filter_size[0],
                                            padding='same',
                                            activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(self.num_filters[1],
                                            self.filter_size[1],
                                            padding='same',
                                            activation='relu')
        self.maxpool = tf.keras.layers.MaxPool1D(3, 3, padding='same')
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.bn = tf.keras.layers.BatchNormalization(trainable=self.is_training)
        self.dense = tf.keras.layers.Dense(self.hidden_size,
                                           activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_classes)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv1(x)   #
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.bn(x)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.dense2(x)

        outputs = tf.nn.softmax(x)

        return outputs

def BOW_net():
    pass

def lstm_net():
    pass

def bilstm_net():
    pass

def gru_net():
    pass


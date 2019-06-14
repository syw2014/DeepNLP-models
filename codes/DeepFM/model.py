#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model.py
# PythonVersion: python3.6
# Date    : 2019/5/7 下午9:15
# IDE     : PyCharm

"""Deep FM implementation with tensorflow according to original paper. ref:https://arxiv.org/abs/1703.04247
"""
import tensorflow as tf
import numpy as np


class DeepFM(object):
    def __init__(self, param=None):
        """
        :param param: Dict object, include model and training parameters
        """
        # define model parameters
        self.params = param

    def batch_norm(self, x, train_phase, scope_bn):
        """

        :param x:
        :param train_phase:
        :param scope_bn:
        :return:
        """
        train_phase = tf.constant(train_phase)
        bn_train = tf.contrib.layers.batch_norm(
            x,
            decay=0.995, center=True, scale=True, updates_collections=None, is_training=True,
            reuse=None, trainable=True, scope=scope_bn)

        bn_inference = tf.contrib.layers.batch_norm(
            x,
            decay=0.995, center=True, scale=True, updates_collections=None, is_training=False,
            reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def inference(self, values, indices, params, is_training=False):
        """
        Portion of the compute graph that takes input and convert it to Y output logit
        :param values:
        :param indices:
        :param params
        :param is_training:
        :return:
        """
        # Notes,
        # values = inputs['values']
        # indices = inputs['indices']

        # l2 regularization
        l2_weight_loss = None

        # ------------ define variables ------------
        # This was the sparse feature to dense feature embedding equal full concetion
        # embedding shape= [feature_size, emb_size], feature_size was the total feature in dataset
        feature_embedding = tf.Variable(initial_value=tf.random_normal(
            shape=[params.feat_size, params.embedding_size], mean=0, stddev=0.1),
            name='feature_embedding',
            dtype=tf.float32)

        # Sparse feature to fc of FM layer Addition unit, equal other embedding
        # shape=[feature_size, 1]
        feature_bias = tf.Variable(initial_value=tf.random_uniform(
            shape=[params.feat_size, 1], minval=0.0, maxval=1.0),
            name='feature_bias',
            dtype=tf.float32)

        # Hidden layer
        layers = dict()
        num_layers = len(params.hidden_layers)
        # After embedding lookup, each field embedding dim was `embedding size`, total dimension was field*dim
        input_size = params.field_size * params.embedding_size
        # global normal: stddev = sqrt(2.0/(fan_in + fan_out))
        glorot = np.sqrt(2.0 / (input_size + params.hidden_layers[0]))

        layers['layer_0'] = tf.Variable(initial_value=tf.random_normal(
            shape=[input_size, params.hidden_layers[0]], mean=0, stddev=glorot),
            name='layer_0',
            dtype=tf.float32)
        layers['bias_0'] = tf.Variable(initial_value=tf.random_normal(
            shape=[1, params.hidden_layers[0]], mean=0, stddev=glorot),
            name='bias_0',
            dtype=tf.float32)

        for i in range(1, num_layers):
            glorot = np.sqrt(2.0 / (params.hidden_layers[i-1] + params.hidden_layers[i]))
            layers['layer_%d' % i] = tf.Variable(initial_value=tf.random_normal(
                shape=[params.hidden_layers[i-1], params.hidden_layers[i]], mean=0, stddev=glorot),
                name="layer_%d" % i,
                dtype=tf.float32)
            layers['bias_%d' % i] = tf.Variable(initial_value=tf.random_normal(
                shape=[1, params.hidden_layers[i]], mean=0, stddev=glorot),
                name="bias_%d" % i,
                dtype=tf.float32)

        #  Output layer Projection, output was one dim
        # final dim
        fm_size = params.field_size + params.embedding_size
        final_size = fm_size + params.hidden_layers[-1]
        glorot = np.sqrt(2.0 / (final_size + 1))
        output_proj = tf.Variable(initial_value=tf.random_normal(
            shape=[final_size, 1], mean=0, stddev=glorot),
            name='projection',
            dtype=tf.float32)
        output_bias = tf.Variable(tf.constant(0.01),
                                  name='output_bias',
                                  dtype=tf.float32)

        # -------------- build network -----------------
        # firstly, embedding
        # Sparse feature -> dense embedding
        # input shape: [None, ids_length], table shape: [feature_size, emb_size]
        # output shape: [None, field_size , embedding_size]
        origin_embedding = tf.nn.embedding_lookup(feature_embedding, ids=indices)
        # input shape: [None, values_fields], output shape: [None, field_size, 1]
        feat_value_reshape = tf.reshape(tensor=values, shape=[-1, params.field_size, 1])

        # --------------- FM Parts -----------------------
        # order-1= <W,X> in equation(2)
        # input shape: [None, ids_length], table shape: [feature_size]
        # output shape: [None, field_size, 1]
        y_first_order = tf.nn.embedding_lookup(feature_bias, ids=indices)
        w_mul_x = tf.multiply(y_first_order, feat_value_reshape)
        # input shape: [None, field_size, 1], output shape: [None, field_size]
        y_first_order = tf.reduce_sum(w_mul_x, axis=2)
        # TODO, add dropout
        y_first_order = tf.nn.dropout(y_first_order, params.fm_dropout_keep)

        # order-2= sum_i* sum_j <V_i, V_j>x_i * x_j
        # out shape: [None, field_size, emb_size]
        embeddings = tf.multiply(origin_embedding, feat_value_reshape)

        # sum_square_part ,sum before square
        # shape: [None, emb_size]
        summed_feature_emb = tf.reduce_sum(embeddings, axis=1)
        summed_square = tf.square(summed_feature_emb)

        squared_emb = tf.square(embeddings)
        # [None, emb_size]
        squared_summed = tf.reduce_sum(squared_emb, axis=1)

        # order-2
        y_second_order = 0.5 * tf.subtract(summed_square, squared_summed)
        # TODO, add dropout
        y_second_order = tf.nn.dropout(y_second_order, params.fm_dropout_keep)

        # --------------- Deep Parts -----------------------
        # shape: [None, field*emb]
        y_deep = tf.reshape(origin_embedding, shape=[-1, input_size])
        # weights_loss = None
        # calculate l2 weights loss
        weights_loss = tf.contrib.layers.l2_regularizer(scale=params.l2_reg)(output_proj)
        # hidden layers
        for i in range(len(params.hidden_layers)):
            y_deep = tf.add(tf.matmul(y_deep, layers['layer_%d' % i]), layers['bias_%d' % i])
            if is_training:
                # TODO, add batch norm
                # bn_train = tf.contrib.layers.batch_norm(
                #     y_deep,
                #     decay=0.995, center=True, scale=True, updates_collections=None, is_training=True,
                #     reuse=None, trainable=True, scope='bn')
                #
                # bn_inference = tf.contrib.layers.batch_norm(
                #     y_deep,
                #     decay=0.995, center=True, scale=True, updates_collections=None, is_training=False,
                #     reuse=None, trainable=True, scope='bn')
                # y_deep = tf.cond(is_training, lambda: bn_train, lambda: bn_inference)

                y_deep = self.batch_norm(y_deep, train_phase=is_training, scope_bn='bn_%d' % i)

            y_deep = tf.nn.relu(y_deep)
            # TODO, add dropout
            y_deep = tf.nn.dropout(y_deep, params.dropout_keep_prob)
            weights_loss += tf.contrib.layers.l2_regularizer(scale=params.l2_reg)(layers['layer_%d' % i])

        #  output
        concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
        logits = tf.add(tf.matmul(concat_input, output_proj), output_bias)
        # output = tf.nn.sigmoid(output)

        return logits, weights_loss

    def loss(self, logits, labels, weight_loss):
        """
        Calculate losses for DeepFM model.
        :param logits: output of network
        :param labels: labels tensor, shape=[None, 1]
        :param weight_loss: Only used for regularization
        :return: loss tensor
        """
        # here use logloss
        outputs = tf.nn.sigmoid(logits)
        # TODO, change loss functions, here to use logloss
        print('debug label shape: ', labels.get_shape(), outputs.get_shape())
        loss = tf.losses.log_loss(labels, outputs)
        loss += weight_loss

        return loss

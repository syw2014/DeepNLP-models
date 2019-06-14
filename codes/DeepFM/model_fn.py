#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : model_fn.py
# PythonVersion: python3.6
# Date    : 2019/6/6 上午11:02
# IDE     : PyCharm
"""Build model helper functions"""

import tensorflow as tf


def build_model_spec(mode, model, inputs, params, reuse=False):
    """
    Build model specification function
    :param mode: (String), to indicate application run in which pattern <train,eval, inference>
    :param model: (Object), defined model object
    :param inputs: (Dict), which contains infos model need, may like {'input_x': x_tensor, 'input_y': label_tensor,...}
    :param params: (Object), just like dict defined in util_tools.py
    :param reuse: (bool), whether to reuse variables or not
    :return: (Dict) model specification
    """
    model_spec = inputs
    is_training = (mode == 'train')
    feat_values = inputs['values']
    feat_indices = inputs['indices']

    with tf.variable_scope('model', reuse=reuse):
        logits, weight_loss = model.inference(values=feat_values, indices=feat_indices, params=params,
                                              is_training=is_training)
        score = tf.nn.sigmoid(logits)
        prediction = tf.cast(tf.argmax(score, axis=1), tf.float32)

    if mode == 'train':
        labels = inputs['labels']
        loss = model.loss(logits, labels, weight_loss)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels, prediction), tf.float32))

        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)

        # metrics and summaries
        # Metrics for evaluation using tf.metrics (average for whole dataset)
        with tf.variable_scope('metrics'):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=prediction),
                'loss': tf.metrics.mean(loss)
            }
        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_init_op = tf.variables_initializer(metric_variables)

        # summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)

        model_spec['loss'] = loss
        model_spec['accuracy'] = acc
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()
        model_spec['train_op'] = train_op
    elif mode == 'eval':
        labels = inputs['labels']
        loss = model.loss(logits, labels, weight_loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, prediction), tf.float32))

        # metrics and summaries
        # Metrics for evaluation using tf.metrics (average for whole dataset)
        with tf.variable_scope('metrics'):
            metrics = {
                'accuracy': tf.metrics.accuracy(labels=labels, predictions=prediction),
                'loss': tf.metrics.mean(loss)
            }
        # Group the update ops for the tf.metrics
        update_metrics_op = tf.group(*[op for _, op in metrics.values()])

        # Get the op to reset the local variables used in tf.metrics
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='metrics')
        metrics_init_op = tf.variables_initializer(metric_variables)

        # summaries for training
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)

        model_spec['loss'] = loss
        model_spec['accuracy'] = accuracy
        model_spec['metrics_init_op'] = metrics_init_op
        model_spec['metrics'] = metrics
        model_spec['update_metrics'] = update_metrics_op
        model_spec['summary_op'] = tf.summary.merge_all()

    variable_init_op = tf.group(*[tf.global_variables_initializer(),
                                  tf.local_variables_initializer(),
                                  tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op

    model_spec['prediction'] = prediction
    model_spec['score'] = score

    return model_spec

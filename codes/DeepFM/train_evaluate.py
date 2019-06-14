#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train_evaluate.py
# PythonVersion: python3.6
# Date    : 2019/6/6 上午11:52
# IDE     : PyCharm

"""Define train and evaluate graph and sess"""
import logging
import tensorflow as tf
from util_tools import save_dict_json
from tqdm import trange
import numpy as np
import os


def train_sess(sess, model_spec, num_steps, writer, params):
    """
    Define train graph
    :param sess: tf.Session
    :param model_spec: (Dict) which contains graph operations or nodes needed for model training
    :param num_steps: number of train steps
    :param writer: tf.summary writer
    :param params: (Object), Parameters of models and datasets
    :return:
    """

    # Step1, Get relevant graph operations or nodes needed for training
    train_op = model_spec['train_op']
    loss = model_spec['loss']
    update_metrics = model_spec['update_metrics']  # loop over all dataset
    summary_op = model_spec['summary_op']
    metrics = model_spec['metrics']
    global_step = tf.train.get_or_create_global_step()  # get global train step

    # Step2, initialize variables
    sess.run(model_spec['metrics_init_op'])     # metrics op
    sess.run(model_spec['iterator_init_op'])    # iterator op

    # Step3, loop train steps
    # use tqdm trange as process bar
    t = trange(num_steps)
    for i in t:
        # write summary after summary_steps
        if i % params.save_summary_steps == 0:
            _, loss_val, _, summary_val, step_val = sess.run([train_op, loss, update_metrics,
                                                                           summary_op, global_step])
            writer.add_summary(summary_val, step_val)
        else:
            _, _, loss_val = sess.run([train_op, update_metrics, loss])
            t.set_postfix(loss='{:05.3f}'.format(loss_val))

    # Step4 print metrics
    metric_val_tensor = {k: v[0] for k, v in metrics.items()}
    metric_vals = sess.run(metric_val_tensor)
    metric_vals_str = ' ; '.join('{}: {:05.3f}'.format(k,v) for k, v in metric_vals.items())
    logging.info('- Train Metrics: '+ metric_vals_str)


def eval_sess(sess, model_spec, num_steps, writer=None, params=None):
    """
    Define evaluate graph
    :param sess: tf.Session
    :param model_spec: (Dict) which contains graph operations or nodes needed for model training
    :param num_steps: number of evaluate steps
    :param writer: tf.summary writer, will create new if none
    :param params: (Object), Parameters of models and datasets
    :return:
    """

    # Step1, get relevant graph operations or nodes for model evaluation
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_or_create_global_step()

    # Step2, initialize operations or nodes
    sess.run(model_spec['iterator_init_op'])    # dataset iterator op
    sess.run(model_spec['metrics_init_op'])     # metrics op

    # Step3, loop evaluate steps to calculate metrics
    for _ in range(num_steps):
        sess.run(update_metrics)

    # Step 4, get metrics values
    metrics_val_tensors = {k: v[0] for k, v in eval_metrics.items()}
    metrics_vals = sess.run(metrics_val_tensors)
    metrics_vals_str = ' ; '.join('{}: {:05.3f}'.format(k, v) for k, v in metrics_vals.items())
    logging.info("- Eval Metrics:" + metrics_vals_str)

    # Step5, write summary
    if writer is not None:
        global_steps_val = sess.run(global_step)
        for tag, val in metrics_vals.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
            writer.add_summary(summ, global_steps_val)
    return metrics_vals


def evaluate(model_spec, model_dir, params, restore_from):
    """

    :param model_spec:
    :param model_dir:
    :param params:
    :param restore_from:
    :return:
    """
    # Step1, initialize tf.train.Saver()
    saver = tf.train.Saver()

    # Step2, create session and initialize variables
    with tf.Session() as sess:
        # initialize all variables
        sess.run(model_spec['variable_init_op'])

        # Step3, restore models from `restore_from`
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Step4, evaluate models with eval_sess
        num_steps = (params.dev_size + params.batch_size - 1) // params.batch_size
        metrics = eval_sess(sess, model_spec, num_steps)

        # Step5, write metric result to file
        metrics_names = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir, "metrics_eval_{}.json".format(metrics_names))
        save_dict_json(metrics, save_path)


def train_evaluate(train_model_spec, eval_model_spec, model_dir, params, restore_from=None):
    """
    Train the model and evaluate model in every epoch.
    :param train_model_spec:(dict), contains graph operations or nodes needed for model train
    :param eval_model_spec: (dict), contains graph operations or nodes needed for model evaluate
    :param model_dir: (string), the path where to save model
    :param params: (Object) Parameters, contains hyperparameters and model parameters
    :param restore_from: (String), directory or file containing weights to restore the graph
    :return:
    """
    # Step1, Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver()
    best_saver = tf.train.Saver(max_to_keep=1)  # only keep 1 best checkpoints (best on eval)
    begin_at_epoch = 0  # record last trained epoch

    # Step2, create session and initialize variables
    with tf.Session() as sess:
        sess.run(train_model_spec['variable_init_op'])
        sess.run(tf.global_variables_initializer())

        # Step3, reload previous trained model or not
        if restore_from is not None:
            logging.info('Restoring parameters from {}'.format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # Step4, define summary writer
        train_writer = tf.summary.FileWriter(os.path.join(model_dir, 'train_summaries'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir, 'eval_summaries'), sess.graph)

        # Step5, train epochs
        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch, begin_at_epoch + params.epochs):
            # Step6, train sess
            num_steps = (params.train_size + params.batch_size - 1) // params.batch_size
            train_sess(sess, train_model_spec, num_steps, train_writer, params)

            # Step7, save weights
            last_save_path = os.path.join(model_dir, 'last_weights', 'after-epoch')

            last_saver.save(sess,  last_save_path, global_step=epoch+1)

            # Step8, evaluate model
            num_steps = (params.dev_size + params.batch_size - 1) // params.batch_size
            metrics = eval_sess(sess, eval_model_spec, num_steps, eval_writer)

            # if best eval, best save to path
            eval_acc = metrics['accuracy']
            if eval_acc >= best_eval_acc:
                best_eval_acc = eval_acc

                # Step9, save best model
                best_save_path = os.path.join(model_dir,  'best_weights',  'after-epoch')
                best_save_path = best_saver.save(sess, best_save_path, global_step=epoch+1)

                logging.info('- Found new best accuracy , saving in {}'.format(best_save_path))
                best_weight_json = os.path.join(model_dir, 'metrics_eval_last_weights.json')
                save_dict_json(metrics, best_weight_json)

        # save latest eval metrics in a json file in the model dir
        latest_json_path = os.path.join(model_dir, 'metrics_eval_last_weights.json')
        save_dict_json(metrics, latest_json_path)




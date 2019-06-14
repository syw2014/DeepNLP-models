#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train.py
# PythonVersion: python3.6
# Date    : 2019/6/7 下午1:52
# IDE     : PyCharm
"""Train entrance for model."""

import tensorflow as tf
from util_tools import Params, set_logger
from input_fn import load_dataset_from_text, input_fn
from model_fn import build_model_spec
from train_evaluate import train_evaluate, evaluate
from model import DeepFM

import pickle
import argparse
import logging
import os


# Step1, define parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/root/data/research/data/product/rs/kaggle_DAC/results',
                    help='model directory')
parser.add_argument('--data_dir', default='/root/data/research/data/product/rs/kaggle_DAC/processed',
                    help='directory containing dataset and params.json')
parser.add_argument('--restore_from', default=None, help='optional, directory containing weights to reload')


def main():
    tf.set_random_seed(230)
    args = parser.parse_args()
    param_path = os.path.join(args.data_dir, 'params.json')
    assert os.path.isfile(param_path), 'No <dataset_params.json> found in path: {}'.format(args.data_dir)
    # load parameters
    params = Params(param_path)
    params.buffer_size = params.train_size
    params.hidden_layers = [int(x) for x in params.hidden_layers.split(',')]

    set_logger(os.path.join(args.model_dir, 'train.log'))
    params.print()

    # Step2, create tf.dataset
    logging.info('Create train and eval dataset...')
    train_file = os.path.join(args.data_dir, 'train1.tsv')
    eval_file = os.path.join(args.data_dir, 'dev1.tsv')
    train_dataset = load_dataset_from_text(train_file, params)
    eval_dataset = load_dataset_from_text(eval_file, params)

    # Step3, create train and eval iterator over two datset
    train_inputs = input_fn(train_dataset, params, 'train')
    eval_inputs = input_fn(eval_dataset, params, 'eval')
    logging.info('Completed create input pipeline!')

    # Step4, define model
    logging.info('Create model...')
    model = DeepFM()

    # Step5, build model specification
    train_model_spec = build_model_spec('train', model, train_inputs, params, reuse=False)
    # If you want to only run evaluate you should set reuse=False.
    eval_model_spec = build_model_spec('eval', model, eval_inputs, params, reuse=True)
    logging.info('Create train and eval model specification completed!')

    # Step6, train and evaluate model
    logging.info('Start training for {} epochs'.format(params.epochs))
    train_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, args.restore_from)

    # Step7, save model
    with open(args.model_dir + '/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # evaluate model
    # evaluate(eval_model_spec, args.model_dir, params, args.restore_from)
 

if __name__ == '__main__':
    main()
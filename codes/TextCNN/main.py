#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : train.py
# PythonVersion: python3.6
# Date    : 2019/10/19 10:08
# Software: PyCharm

import tensorflow as tf
import argparse
from modelling import TextCNN
from create_vocab import Vocab
from data_utils import create_dataset_with_tf, create_single_input
import pandas as pd
import numpy as np
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", default="../data/short_doc_classification/doc_valid.txt", type=str,
                    help="train data file")
parser.add_argument("--valid_data", default="../data/short_doc_classification/doc_valid.txt", type=str,
                    help="validation data file")
parser.add_argument("--predict_data", default="../data/short_doc_classification/doc_valid.txt", type=str,
                    help="predict  data file")
parser.add_argument("--stopwords", default="", type=str,
                    help="stop words file for create vocabulary")
parser.add_argument("--model_dir", default="../result/", type=str,
                    help="trained model directory")
parser.add_argument("--tensorboard_dir", default="../result/", type=str,
                    help="tensorboard files directory")
parser.add_argument("--mode", default="train", type=str,
                    help="set which pattern the program will run in train/test/inference")

parser.add_argument("--vocab_size", default=40000, type=int,
                    help="vocabulary size")
parser.add_argument("--max_seq_length", default=20, type=int,
                    help="the maximum sentence length")
parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size, how many samples in one batch")
parser.add_argument("--num_classes", default=7, type=int,
                    help="how many labels in dataset")
parser.add_argument("--epochs", default=1, type=int,
                    help="how many rounds want to train")

# training parameters
parser.add_argument("--learning_rate", default=0.001, type=float,
                    help="learning rate")
parser.add_argument("--embedding_dim", default=32, type=float,
                    help="word embedding dimension")
parser.add_argument("--num_filters", default='256,32', type=str,
                    help="number filters in different convolution layers")
parser.add_argument("--filter_sizes", default="3,2", type=str,
                    help="filter size used in different convolution layers")
parser.add_argument("--dropout_rate", default=0.3, type=float,
                    help="dropout rate in dropout layer")
parser.add_argument("--hidden_size", default=256, type=int,
                    help="number of hidden units used in dense layer")

args = parser.parse_args()


@tf.function
def loss_object(y_pred, labels):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=y_pred)
    loss = tf.reduce_mean(loss)
    return loss


def evaluate(valid_dataset, num_samples, eval_batch, model):
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    steps_per_valid = num_samples // eval_batch
    loss = 0.0
    cnt = 0
    for (ids, (inputs, labels)) in enumerate(valid_dataset.take(steps_per_valid)):
        y_pred = model.predict(inputs)
        loss += loss_object(y_pred, labels)
        sparse_categorical_accuracy.update_state(y_true=labels, y_pred=y_pred)
        cnt += 1
    print("loss: ", loss.numpy() / cnt)
    loss_val = loss.numpy() / cnt
    return sparse_categorical_accuracy.result().numpy(), loss_val, sparse_categorical_accuracy.result(), loss


def predict(predict_dataset, num_samples, predict_batch, model):
    """
    Result predict
    :param predict_dataset: input filename
    :param num_samples:
    :param eval_batch:
    :param model:
    :return:
    """
    steps_per_valid = num_samples // predict_batch 
    #steps_per_valid = 1
    for (ids, (inputs, labels)) in enumerate(predict_dataset.take(steps_per_valid)):
        y_pred = model.predict(inputs)
        # extract the biggest probability index
        # predictions = tf.math.argmax(y_pred, axis=1)
        # probs = tf.gather(y_pred, predictions, axis=1)
        predictions = tf.math.top_k(y_pred, k=1)
        values, indics = predictions
        #print(values.numpy(), indics.numpy())
        
        # extract predictions
        probs = values.numpy()
        indices = indics.numpy()
        for i, p in enumerate(probs):
            prob = p[0]
            label_idx = indices[i][0]
            print(prob, label_idx)

    return None

def predict_single(inputs, model):
    """
    Predict single sample.
    """
    y_pred = model.predict(np.array([inputs,]))
    predictions = tf.math.top_k(y_pred, k=1)
    probs, indics = predictions
    probs = probs.numpy()[0]
    indics = indics.numpy()[0]
    res = {}
    #print("predict result: ", probs, indics)
    for i, p in enumerate(probs):
        res['label'] = indics[i]
        res['probability'] = p
    #print(probs.numpy(), indics.numpy())
    print(res)

    return res


def train_op(model, optimizer, train_dataset, eval_dataset, num_train_samples,
             num_eval_samples,checkpoint, checkpoint_prefix, tensorboard_dir):
    """

    :param model:
    :param optimizer:
    :param train_dataset:
    :param eval_dataset:
    :param num_train_samples:
    :param num_eval_samples:
    :param checkpoint:
    :param checkpoint_prefix:
    :return:
    """

    # @tf.function    # convert to static graph for speed
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            y_pred = model(inputs)
            # loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=y_pred)
            # loss = tf.reduce_mean(loss)
            loss = loss_object(y_pred, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, y_pred

    # train
    steps_per_epoch = num_train_samples // args.batch_size
    print_template = "Epoch:{} Batch:{} Train Loss:{:.4f} Train Accuracy:{:.4f}"
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    # create summary
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)

    for epoch in range(args.epochs):
        start = time.time()
        loss = 0
        # train one epoch
        for (batch_index, (inputs, labels)) in enumerate(train_dataset.take(steps_per_epoch)):
            loss, predictions = train_step(inputs, labels)
            train_accuracy.update_state(y_true=labels, y_pred=predictions)
            # write to summary
            with summary_writer.as_default():
                tf.summary.scalar("train_loss", loss, step=epoch*steps_per_epoch+batch_index)
                tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch*steps_per_epoch+batch_index)
            if batch_index % 10 == 0:
                print(print_template.format(epoch+1, batch_index, loss.numpy(),
                                            train_accuracy.result().numpy()))

        train_accuracy.reset_states()
        acc,eval_loss, acc_tensor,loss_tensor = evaluate(eval_dataset, num_eval_samples, 64, model)
        # write eval info to summary
        with summary_writer.as_default():
            tf.summary.scalar("eval_loss", loss_tensor, step=epoch)
            tf.summary.scalar("eval_acc", acc_tensor, step=epoch)
        print("After Epoch {} Train Loss {:.4f} cost time: {}s Eval acc {:.4f} Eval Loss:{:.4f}".
                format(epoch+1, loss, (time.time() - start), acc, eval_loss))

        # saving checkpoint model every 2 epochs
        if (epoch+1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


def main():
    # Create vocabulary Object
    vocab = Vocab(stopwords_file=args.stopwords,
                  vocab_size=args.vocab_size)
    vocab.load()

    # create model
    model = TextCNN(args.vocab_size,
                    args.embedding_dim,
                    args.max_seq_length,
                    args.num_filters,
                    args.filter_sizes,
                    args.num_classes,
                    args.dropout_rate,
                    args.hidden_size,
                    is_training=True)

    # create train process
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    # define checkpoint saver
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_dir = args.model_dir + '/checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    if args.mode == "train":
        train_dataset, num_train_samples = create_dataset_with_tf(args.train_data, vocab, args.epochs,
                                args.batch_size,
                                args.max_seq_length,
                                mode="train")
        eval_dataset, num_eval_samples = create_dataset_with_tf(args.valid_data, vocab, 1,
                                args.batch_size,
                                args.max_seq_length,
                                mode="test")
        # train model
        train_op(model, optimizer, train_dataset, eval_dataset, num_train_samples, num_eval_samples,
                    checkpoint, checkpoint_prefix, args.tensorboard_dir)

    elif args.mode == "test":
        eval_dataset, num_eval_samples = create_dataset_with_tf(args.valid_data, vocab, 1,
                                128,
                                args.max_seq_length,
                                mode="test")
        # reload model
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        acc, loss = evaluate(eval_dataset, num_eval_samples, 128, model)
        print("Eval acc {:.4f} Eval Loss:{:.4f}".format(acc, loss))

    elif args.mode == "inference":
        pred_dataset, num_pred_samples = create_dataset_with_tf(args.predict_data, vocab, 1,
                                128,
                                args.max_seq_length,
                                mode="predict")
        # reload model
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        predict(pred_dataset, num_pred_samples, 128, model)

    # reload model
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # acc = evaluate(train_dataset, num_train_samples, 64, model)
    # print("Eval acc {:.4f}".format(acc))
    #docs = "燃【英雄联盟CG动画】终有一日我的天才定会得到理解，戏命师-烬，高清"
    #inputs = create_single_input(docs, vocab, args.max_seq_length)
    #predict(train_dataset, num_train_samples, 64, model)
    #predict_single(inputs, model)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : gen_vocab.py
# PythonVersion: python3.5
# Date    : 2019/1/15 14:59
# Software: PyCharm

"""This file mains to complete this functions,1) document segmentation 2) create vocabulary
Notes:
    1. use multi-thread to segment
    2. create vocabulary with train/dev/test files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jieba
import time
import multiprocessing

from collections import Counter
import argparse
from tqdm import tqdm
import pandas as pd
import random
import re

jieba.initialize()

# parser = argparse.ArgumentParser()
# # Define hyper-parameters, you should set the value based on your own projects
# parser.add_argument('--most_common', type=int, default=0, help='most common number of words you want to use')
# parser.add_argument('--min_count', type=int, default=0, help='The minimum term frequency of words thread')
# parser.add_argument('--add_unknown', type=bool, default=True, help='Treat ignored wors as unknown')
# args = parser.parse_args()

START_WORD = '<S>'
END_WORD = '</S>'


def parallel_segment(text, seg_method=jieba):
    """
    Text segment
    :param text: input text string
    :return: token list
    """
    tokens = list(jieba.cut(text.strip()))

    return tokens


def tokenize(lines):
    """
    Use multi-thread to segment
    :param lines: input text iterator
    :return:
    """
    print('Debug==>{}'.format(len(lines)))
    # create thread pool
    s_time = time.time()
    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.7))
    result = pool.map(parallel_segment, lines)

    pool.close()
    pool.join()
    e_time = time.time()
    print("Segment {} lines cost time:{}s".format(len(lines), e_time - s_time))

    return result


def test_multi_thread():
    filename = '/data/research/data/doc-classify/doc.txt'
    lines = open(filename, 'r').readlines()
    result = tokenize(lines)
    with open('doc_tokens.txt', 'w') as f:
        for tokens in result:
            f.write("\t".join(tokens))
            f.write('\n')


class WordCounter(object):
    """Create word counter to create vocabulary."""
    def __init__(self, most_common=None, min_count=None, write_unknown=True, unknown_mark='<UNK>'):
        """
        Word counter object.
        :param most_common: Integer,Choose topk frequent words
        :param min_count: Integer, minimum term frequency
        :param write_unknown: Bool, to write unknow to file
        :param unknown_mark: String, use '<UNK>' to indicate unknow words
        """
        self.most_common = most_common
        self.min_count = min_count or 0
        self.write_unkown = write_unknown
        self.unknown_mark = unknown_mark

        # counter
        self.counter = Counter()
        self.total_tokens = 0

    def add(self, token, count=1):
        """Add token into counter."""
        self.counter[token] += count
        self.total_tokens += 1

    def save(self, filename, most_common=None, min_count=None):
        """
        To save vocabulary to file
        :param filename: output file name
        :param most_common: Integer,Choose topk frequent words
        :param min_count: Integer, minimum term frequency
        :return:
        """
        if not most_common:
            most_common = self.most_common
            if not most_common:
                most_common = len(self.counter)
        if not min_count:
            min_count = self.min_count

        with open(filename, 'w') as f:
            reversed_words = []
            reversed_total = 0  # reversed tokens
            # clean words based on min count
            for word, count in self.counter.most_common(most_common):
                if count > min_count:
                    reversed_words.append((word, count))
                    reversed_total += count
            # write unknown to file
            if self.write_unkown:
                unknown_count = self.total_tokens - reversed_total
            else:
                unknown_count = 0
            print("Found total words {}, after remove term frequency bellow {} remain vocabulary size: {}".format(
                self.total_tokens, min_count, reversed_total))

            # write words and term frequency to file
            for word, count in reversed_words:
                if unknown_count > 0:
                    f.write(self.unknown_mark + '\t' + str(unknown_count))
                    f.write('\n')
                    unknown_count = 0
                f.write(word + '\t' + str(count))
                f.write('\n')


# create text clean methods
def filter_duplicate_space(text):
    # remove multiple space only keep one
    return ''.join([x for i, x in enumerate(text) if not (i < len(text) - 1 and not x.strip()
                                                           and not text[i+1].strip())])


def filter(text):
    # filter quota
    text = text.replace("''", '" ').replace("``", '" ')
    text = text.replace('\"', "")
    # remove \n or \r\n in string
    # in windows
    text = re.sub('\r?\n', '', text)
    # in unix
    text = re.sub('\n', '', text)
    text = filter_duplicate_space(text)
    # TODO
    text = text.lower()
    return text


def create_vocab(lines, counter):
    """
    Create vocabulary
    :param lines: line iterator
    :return:
    """
    raw_num = len(counter.counter)
    if len(lines) == 0:
        print('No lines need to add in to vocabulary!')
        return None

    # create words
    # add start word in vocabulary
    counter.add(START_WORD)
    for line in tqdm(lines):
        words = line.strip().split()
        for word in words:
            if len(word.strip()) == 0:
                continue
            counter.add(word)
            # use <NUM> to instead of number
            # if word.isdigit():
            #     counter.add('<NUM>')
    # add stop word in vocabulary
    counter.add(END_WORD)

    print("Add {} word to vocabulary".format(len(counter.counter) - raw_num))


def csv_parser(data_dir):
    """
    Parse zhidao csv file and return texts.
    :param data_dir: input file data dir
    :return: texts and labels
    """

    # load data and split train/dev/test
    dataframe = pd.read_csv(data_dir + '/financezhidao_filter.csv')
    dataset = []
    # remove samples which title and reply was nan
    dataframe = dataframe[dataframe['reply'].notnull()]
    dataframe = dataframe[dataframe['title'].notnull()]

    for i in tqdm(range(len(dataframe))):
        row = dataframe.iloc[i]
        title = row['title'].strip()
        question = row['question'] if not row['question'] else ""
        answer = row['reply'].strip()
        label = row['is_best']

        dataset.append([title+' '+question, answer, str(label)])
    print('Load data from {}/financezhidao_filter.csv, total sample:{}'.format(data_dir, len(dataset)))

    # load another dataset
    with open(data_dir+'/atec_nlp_sim_train.csv', 'r') as f:
        tmp_size = len(dataset)
        for line in tqdm(f.readlines()):
            arr = line.strip().split('\t')
            if len(arr) != 4:
                print('Bad line found:{}'.format(line))
                continue
            dataset.append([arr[1].strip(), arr[2].strip(), arr[3].strip()])
        print('Load data from {}/atec_nlp_sim_train.csv, total sample:{}'.format(data_dir, len(dataset) - tmp_size))

    # split train/dev/test as 7:1:2, and
    random.shuffle(dataset)
    num_samples = len(dataset)
    train_dataset = dataset[: int(0.7*num_samples)]
    dev_dataset = dataset[int(0.7*num_samples): int(0.7*num_samples)+int(0.1*num_samples)]
    test_dataset = dataset[int(0.7*num_samples)+int(0.1*num_samples):]
    print('Dataset split train/dev/test:{}/{}/{}'.format(len(train_dataset), len(dev_dataset), len(test_dataset)))

    # segment and store data
    # TODO, segment with multi-thread
    # train
    with open(data_dir+'/content.txt', 'w') as fout:
        with open(data_dir+'/train.txt', 'w') as f1, open(data_dir+'/train_jieba_seg.txt', 'w') as f2:
            # trick
            write_train = ["\t".join(ele) for ele in train_dataset]
            f1.write("\n".join(write_train))
            print('Start segment train dataset')
            for ele in tqdm(train_dataset):
                q, a, label = ele
                q = filter(q.strip())
                a = filter(a.strip())
                q_tokens = list(parallel_segment(q))
                a_tokens = list(parallel_segment(a))

                f2.write(' '.join(q_tokens) + '\t' + ' '.join(a_tokens) + '\t' + label + '\n')
                fout.write(' '.join(q_tokens) + '\n' + ' '.join(a_tokens) + '\n')

        #
        with open(data_dir+'/dev.txt', 'w') as f1, open(data_dir+'/dev_jieba_seg.txt', 'w') as f2:
            # trick
            write_dev = ["\t".join(ele) for ele in dev_dataset]
            f1.write("\n".join(write_dev))
            print('Start segment dev dataset')
            for ele in tqdm(dev_dataset):
                q, a, label = ele
                q = filter(q.strip())
                a = filter(a.strip())
                q_tokens = list(parallel_segment(q))
                a_tokens = list(parallel_segment(a))

                f2.write(' '.join(q_tokens) + '\t' + ' '.join(a_tokens) + '\t' + label + '\n')
                fout.write(' '.join(q_tokens) + '\n' + ' '.join(a_tokens) + '\n')

        with open(data_dir+'/test.txt', 'w') as f1, open(data_dir+'/test_jieba_seg.txt', 'w') as f2:
            # trick
            write_test = ["\t".join(ele) for ele in test_dataset]
            f1.write("\n".join(write_test))
            print('Start segment test dataset')
            for ele in tqdm(test_dataset):
                q, a, label = ele
                q = filter(q.strip())
                a = filter(a.strip())
                q_tokens = list(parallel_segment(q))
                a_tokens = list(parallel_segment(a))

                f2.write(' '.join(q_tokens) + '\t' + ' '.join(a_tokens) + '\t' + label + '\n')
                fout.write(' '.join(q_tokens) + '\n' + ' '.join(a_tokens) + '\n')


def main():
    # segment
    # words_result = parallel_segment(lines)
    # load train data and create vocabulary
    data_dir = '/data/research/data/textMatch/'
    # ------------------Step 1----------------------
    # parse and tokenize
    s_time = time.time()
    csv_parser(data_dir)
    e_time = time.time()
    print("Parse file complete total time:{}s".format(e_time - s_time))

    counter = WordCounter(min_count=5)
    # load segment file and create vocab
    filename = data_dir + '/content.txt'
    s_time = time.time()
    print("Create vocabulary from {}".format(filename))
    lines = open(filename, 'r').readlines()
    create_vocab(lines, counter)
    e_time = time.time()
    print("Create vocabulary complete total words:{}, cost time:{}".format(len(counter.counter),e_time - s_time))
    counter.save(data_dir + 'vocab_word.txt')


if __name__ == "__main__":
    main()
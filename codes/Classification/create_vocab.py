#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : create_vocab.py
# PythonVersion: python3.6
# Date    : 2019/10/19 10:15
# Software: PyCharm
"""Create vocabulary"""


from collections import Counter
import jieba
import jieba.analyse as analyse
import json
from tqdm import tqdm
try:
    from opencc import OpenCC
except ImportError:
    print("Should run `pip install opencc-python-reimplemented` to install opencc package")

# traditional to simplified chinese
t2s = OpenCC('t2s')


class Vocab:
    def __init__(self, stopwords_file=None, vocab_size=None, vocab_dir=None):
        self.vocab_size = vocab_size
        self.vocab_dir = None

        self.counter = Counter()
        self.vocab = {}
        self.reverse_vocab = {}
        self.labels = {}
        if stopwords_file is not None:
            self.stopwords = [line.strip() for line in open(stopwords_file, encoding='utf-8').readlines()
                     if not line.startswith("#")]
        else:
            self.stopwords = []

    def create_count(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            labels = set()
            for line in tqdm(f.readlines()):
                arr = line.strip().split('\t')
                if len(arr) != 2:
                    continue
                labels.add(arr[0].strip())
                doc = arr[1].strip().lower()
                doc = t2s.convert(doc)

                # TODO, use jieba to extract key words based on tf-ids
                # tokens = analyse.extract_tags(doc, topK=50, withWeight=False, allowPOS=())
                # TODO, remove special chars
                tokens = list(jieba.cut(doc))
                tokens = [w for w in tokens if w not in self.stopwords]
                if len(tokens) == 0:
                    continue
                self.counter.update(tokens)
        self.labels = dict(zip(labels, range(len(labels))))

        print("Found {} tokens and {} labels in {}".format(len(self.counter), len(self.labels), filename))

    def gen_vocab(self):
        """
        Create vocabulary
        :return:
        """
        words = self.counter.most_common(self.vocab_size)
        words = [w for (w, c) in words]

        self.vocab = dict(zip(words, range(len(words))))
        self.vocab['UNK'] = len(self.vocab)
        print("Created Vocabulary size was {}".format(len(self.vocab)))

    def save(self):
        """
        Write vocabulary to files
        :return:
        """
        vocab_file = './vocab.json' if self.vocab_dir is None else self.vocab_dir + '/vocab.json'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        label_file = './labels.json' if self.vocab_dir is None else self.vocab_dir + 'labels.json'
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False, indent=4)

    def load(self):

        vocab_file = './vocab.json' if self.vocab_dir is None else self.vocab_dir + '/vocab.json'
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        label_file = './labels.json' if self.vocab_dir is None else self.vocab_dir + 'labels.json'
        with open(label_file, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)

        print("Load vocabulary size: {}, labels: {}".format(len(self.vocab), len(self.labels)))
        print(self.labels)

    def seq_to_ids(self,sequence):
        # token_ids = [self.vocab[x] if x in self.vocab else self.vocab['UNK'] for x in sequence.lower()]
        token_ids = [self.vocab[x] for x in sequence.lower() if x in self.vocab]
        return token_ids

    def doc_to_ids(self, doc, segment=True, remove_stopwords=True):
        tokens = doc
        if segment:
            tokens = list(jieba.cut(doc.lower()))
        # Note, here we do not instead words not in vocabulary
        tokens = [w for w in tokens if w not in self.stopwords]
        ids = [self.vocab[w] for w in tokens if w in self.vocab]
        return ids

if __name__ == '__main__':
    data_dir = '../data/short_doc_classification/'
    filename = data_dir + 'doc_valid.txt'
    vocab_size = 40000
    vocab = Vocab(stopwords_file='../stopwords.txt',
                  vocab_size=vocab_size)

    # vocab.create_count(filename)
    # vocab.create_count(data_dir+'doc_train.txt')
    # vocab.gen_vocab()
    # vocab.save()
    #vocab.load()

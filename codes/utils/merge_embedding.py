#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : merge_embedding.py
# PythonVersion: python3.5
# Date    : 2019/1/25 15:58
# Software: PyCharm

"""Convert Glove embedding vectors file to numpy format.Here we only add Glove model process. word2vec model will
be added in future.
Notes:
    1. Unknown words we assign it's embedding with a random value
    2. According to words with high term frequency but not in vocabulary, we also assign it's embedding vector with
        random value.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', type=str, default='vocab.txt', help='input vocabulary file, '
                                                                        'each line has word and word count')
parser.add_argument('--min_count', type=int, default=20, help='minimum word count')
parser.add_argument('--embedding_file', type=str, default='vectors.txt', help='glove trained vector file, each line has'
                                                                              'word and its specific vector')
parser.add_argument('--output_npy', type=str, default='embedding.npy', help='output numpy file')
parser.add_argument('--output_vocab', type=str, default='word_vocab.txt', help='output words file')
parser.add_argument('--embedding_dim', type=int, default=300, help='embedding size of Glove model')

args = parser.parse_args()


def main():

    # load words from vocabulary, get normalized words and it's count
    lines = open(args.vocab_file, 'r').readlines()
    # words_count = [line.strip().split(' ') for line in lines]
    words_count = []
    original_words, counts = [], []
    for line in lines:
        arr = line.strip().split()
        if len(arr) != 2:
            print(line)
            continue
        original_words.append(arr[0])
        counts.append(arr[1])
    counts = list(map(int, counts))
    original_words = set([w.lower() for w in original_words])
    print('Load word count finished total words:{} toal counts:{}'.format(len(original_words), len(counts)))

    embedding_dict = {}

    # load embedding file, and extract word and it's embedding vector
    dim = args.embedding_dim
    with open(args.embedding_file,'r') as f:
        for i, line in tqdm(enumerate(f.readlines())):
            arr = line.split(' ')
            word = "".join(arr[0: -dim])
            # check vector values
            try:
                vec = list(map(float, arr[-dim:]))
            except Exception:
                print('line in {} found wrong format of embedding value in {}'.format(i, line))
                continue
            if word in original_words:
                embedding_dict[word] = vec
            else:
                embedding_dict[word.lower()] = vec

    # final process to vector and words
    emb_mat = []
    words = []

    # add zero vector for padding, embedding matrix
    emb_mat.append(np.array([0.]*dim))
    # if unknown not in vocabulary, we add random vector for UNK
    if not '<UNK>' in original_words:
        emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(dim)])
        words.append('<UNK>')
        print('Add unknown words UNK into words text')
    for word, count in zip(original_words, counts):
        if word in embedding_dict:
            emb_mat.append(np.array(embedding_dict[word]))
            words.append(word)
        else:
            if count > args.min_count:
                emb_mat.append([np.random.uniform(-0.08, 0.08) for _ in range(dim)])
                words.append(word)

    print('Add all words embedding to emb mat total size:{}, total words:{}'.format(len(emb_mat), len(words)))

    # write to file
    with open(args.output_vocab, 'w') as f1:
        for word in words:
            f1.write(word)
            f1.write('\n')
        np.save(args.output_npy, np.array(emb_mat))


def faiss_test():
    """Use faiss to do fast search"""
    try:
        import faiss
    except Exception:
        print('Faiss package have not install, please check!')
        return

    embedding = np.load(args.output_npy)
    # Note, the data type of embedding must be np.float32 other will cause faiss error
    # <TypeError: in method 'IndexFlat_add', argument 3 of type 'float const>
    embedding = np.float32(embedding)
    index = faiss.IndexFlatIP(args.embedding_dim)
    index.add(embedding)

    # choose five vector to find Top 4 similar vectors
    D, I = index.search(embedding[:5], 4)
    # shape of D= [5, 4] to store the similarity of vectors
    # shape of I = [5, 4] to store the index of top similar vector index
    print(D)
    print(I)


if __name__ == '__main__':
    main()
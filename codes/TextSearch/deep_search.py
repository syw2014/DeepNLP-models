#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : deep_search.py
# PythonVersion: python3.5
# Date    : 2019/1/25 13:45
# Software: PyCharm

"""The main purpose of this application is introduce a new approach for text or query search and get search result.
You can found more introduction in [].
Notes:
    1. This application must run with Faiss package
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from tqdm import tqdm
import jieba
import json

try:
    import faiss
except Exception:
    print('Faiss package have not install, please check!')


class SearchModel(object):
    def __init__(self, filename, embedding_file, emb_dim):
        """
        Instance a search model.
        :param filename: input dataset, which were segment each line with question \t answer \t label, label to show
                        whether the answer is the most related reply.
        :param embedding_file: pre-trained embedding file
        :param emb_dim: embedding dimension
        """
        self.embedding_dict = dict()  # word embedding map
        self.emb_dim = emb_dim
        self.filename = filename
        self.embeding_file = embedding_file

        self.questions_text = []       # list to store question text
        self.question_vec = []
        # TODO, here we only inner product to calculate similarity between two vector
        # self.question_index = faiss.IndexFlatIP(self.emb_dim)
        self.question_index = faiss.IndexFlatL2(self.emb_dim)
        self.answers_text = []         # lis to store answer text
        self.answer_vec = []
        # self.answer_index = faiss.IndexFlatIP(self.emb_dim)
        self.answer_index = faiss.IndexFlatL2(self.emb_dim)

        self.labels = []

    def init(self):
        """To initialize search model."""

        # load word embedding
        s_time = time.time()
        with open(self.embeding_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                arr = line.strip().split()
                word = "".join(arr[:-self.emb_dim])
                # vec = []
                try:
                    vec = list(map(float, arr[-self.emb_dim:]))
                except:
                    print('Bad line {} found in embedding file.'.format(i))
                self.embedding_dict[word.lower()] = vec
            if '<UNK>' not in self.embedding_dict:
                print('Add <UNK> into embdding dict')
                self.embedding_dict['<UNK>'] = [np.random.uniform(-0.08, 0.08) for _ in range(self.emb_dim)]
        e_time = time.time()
        print('Loaded word embedding from file:{} total words:{}, time cost:{}s'.format(
            self.embeding_file, len(self.embedding_dict), (e_time - s_time)))

        # load dataset and parse
        print('Start to parse info parts from dataset {}...'.format(self.filename))
        s_time = time.time()
        with open(self.filename, 'r') as f:
            for line in tqdm(f.readlines()):
                arr = line.strip().split('\t')

                q = arr[0]
                self.questions_text.append("".join(q.split()))
                self.question_vec.append(self.text_to_vec(q.split()))

                a = arr[1]
                self.answers_text.append("".join(a.split()))
                self.answer_vec.append(self.text_to_vec(a.split()))

                self.labels.append(arr[2])
        e_time = time.time()
        print('Parse completed cost time {}s, start to build index with faiss...'.format(e_time - s_time))
        # create index
        # Note, faiss only can process float32 type
        s_time = time.time()
        self.question_index.add(np.array(self.question_vec))
        self.answer_index.add(np.array(self.answer_vec))
        e_time = time.time()
        print('Completed build index time:{}s'.format(e_time - s_time))

    def text_to_vec(self, text, segment=False):
        """
        Convert text to fixed dimension vector
        :param text: input text or word list
        :param segment: bool, to show whether the input need segment or not
        :return: fixed dimension vector
        """
        if segment:
            text = list(jieba.cut(text))

        vec = np.array([0.]*self.emb_dim)
        for w in text:
            if w.lower() in self.embedding_dict:
                vec += self.embedding_dict[w]
            else:
                vec += self.embedding_dict['<UNK>']

        return np.float32(vec) / len(text)

    @staticmethod
    def _index_search(index, vec, topk):
        """
        Search topk most similar vectors in given Faiss index.
        :param index: Faiss builded index
        :param vec: input vector
        :param topk: topk result want to select
        :return:
        """
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        sim, ids = index.search(vec, topk)
        return sim, ids

    def _rank(self, vec_sim, vec_ids, question_sim, question_ids=None):
        """
        To calculate final answer tensor with similar
        :param vec_sim: answer similar tensor
        :param vec_ids: answer id tensor
        :param question_sim: relate question tensor with similar
        :param question_ids: relate question id
        :return: answer text list
        """
        # first we check question number is the same as the answer number
        assert np.shape(vec_sim) == np.shape(vec_ids) and np.shape(vec_sim)[0] == len(question_sim)
        print("Debug question_sim==>{}".format(question_sim))
        # normalize
        qmin, qmax = question_sim.min(), question_sim.max()
        question_sim = (question_sim - qmin) / (qmax - qmin)
        vmin, vmax = vec_sim.min(), vec_sim.max()
        vec_sim = (vec_sim - vmin) / (vmax - vmin)

        # calculate weight
        for i, w in enumerate(question_sim):
            vec_sim[i, :] = vec_sim[i, :] * w

        # result
        ans_score_dict = {}
        shape = np.shape(vec_sim)
        row, column = shape
        for i in range(row):
            for j in range(column):
                id = vec_ids[i, j]
                score = vec_sim[i, j]
                if id not in ans_score_dict:
                    ans_score_dict[id] = score
        # sort by score
        ans_sorted = sorted(ans_score_dict.items(), key=lambda kv:kv[1], reverse=True)
        print("Debug ans_sorted==> ", ans_sorted)
        return ans_sorted

    def question_search(self, vec, question_topk=5):
        """
        Search most related questions
        :param vec: input question vector
        :param question_topk:
        :return:
        """
        sim, qs_ids = self._index_search(index=self.question_index, vec=[vec], topk=question_topk)
        related_ids = qs_ids[0]
        id_result = []
        sim_result = []
        answer_vec = []
        # check question has the right answer
        for i, id in enumerate(related_ids):
            if id > len(self.question_vec):
                continue
            if self.labels[id] == "1":
                id_result.append(id)
                sim_result.append(sim[0][i])
                answer_vec.append(self.answer_vec[id])
        return id_result, sim_result, answer_vec

    def search(self, query, topk=3, question_topk=5):
        """
        Given one query to get it's related answer
        :param query: input query
        :param topk: the number of related answer want to return
        :param question_topk: number question to return most related
        :return: answer text
        """

        result = dict()
        result['query'] = query
        result['answers'] = []
        result['answer_score'] = []

        # step1, convert text to vector
        vec = self.text_to_vec(query, segment=True)
        print('Input query:{} convert to vector completed!'.format(query))
        # step2, get most related questions get the most related question
        # Note, here search query must be a 2-dimension tensor sim is the similarity value tensor with
        # shape=[num_vec,question_topk] qs_ids the id of most related similar vector shape the same as 'sim'
        # sim, qs_ids = self.question_index.search(np.array([vec]), question_topk)
        sim, qs_ids = self._index_search(index=self.question_index, vec=[vec], topk=question_topk)
        print("Found {} most related question!".format(np.shape(sim)))

        # step3, check question has the right answer
        id_result, sim_result, answer_vec = self.question_search(vec, question_topk)
        # second search
        if len(id_result) == 0:
            id_result, sim_result, answer_vec = self.question_search(vec, question_topk+5)
        print('After filter get related question number:{}'.format(len(id_result)))
        if len(id_result) == 0:
            return json.dumps(result)
        # step4, get related answers
        vec_sim, vec_ids = self._index_search(index=self.answer_index, vec=answer_vec, topk=topk)
        # if use L2 distance to calculate similarity, you should convert the result to real sim
        if len(sim_result) != 1:
            sim_result = 1 - np.array(sim_result)
        vec_sim = 1 - vec_sim

        # rank and return results
        ans_sorted = self._rank(vec_sim, vec_ids, sim_result)[:topk]

        for id, score in ans_sorted:
            if id > len(self.answers_text):
                print('No match result found for id:{}'.format(id))
                continue
            text = self.answers_text[id]
            result['answers'].append(text)
            result['answer_score'].append(str(score))
        return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    data_dir = '/data/research/data/textMatch/'
    input_file = data_dir + 'dev_jieba_seg.txt'
    embedding_file = data_dir + 'output/vectors.txt'
    dim = 300
    model = SearchModel(input_file, embedding_file, dim)
    model.init()

    query_list = ["银川百吉大酒店的招商银行叫啥网点", "人脸验证开通花呗"]
    for query in query_list:
        result = model.search(query)
        print(result)
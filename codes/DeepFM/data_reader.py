#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : data_reader.py
# PythonVersion: python3.6
# Date    : 2019/5/15 上午11:20
# IDE     : PyCharm

"""Data Process, here to generate feature dict for dataset"""

import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import os
import time
import json


def chunkify(filename, size=1024 * 1024):
    """Split data into chunks.
    """
    file_end = os.path.getsize(filename)
    with open(filename, 'rb') as f:
        chunk_end = f.tell()
        while True:
            chunk_start = chunk_end
            f.seek(size, 1)
            f.readline()
            chunk_end = f.tell()
            yield chunk_start, chunk_end - chunk_start
            if chunk_end > file_end:
                break


def init(l):
    global lock
    lock = l


class FeatDict(object):
    def __init__(self, dirs, filename, numeric_cols, discrete_cols, ignore_cols, clip_values):
        """

        :param dirs: path of dataset
        :param filename: input dataset file name
        :param numeric_cols: numeric columns
        :param discrete_cols: discrete columns or categorical fields
        :param ignore_cols: ignore columns
        :param clip_values: clipped values for categorical values
        """
        self.dirs = dirs
        self.filename = filename
        self.numeric_cols = numeric_cols
        self.discrete_cols = discrete_cols
        self.ignore_cols = ignore_cols
        self.clip_values = clip_values
        self.feat_dict = {}
        self.total_cnt = 0
        self.integer_norm = {}

    def _parse(self, line):
        """
        Parse each line
        :param line: input line
        :return:
        """
        arr = line.strip().split('\t')
        for col, e in enumerate(arr):
            # ignore cols
            if col in self.ignore_cols:
                continue

            # process numeric cols, numeric columns regard as one index, then only add to dict once
            if col in self.numeric_cols:
                # TODO, integer normalize
                if e != "":
                    val = int(e)
                    if col not in self.integer_norm:
                        self.integer_norm[col] = {'min': val, 'max': val}
                    else:
                        if self.integer_norm[col]['min'] > val:
                            self.integer_norm[col]['min'] = val
                        elif self.integer_norm[col]['max'] < val:
                            self.integer_norm[col]['max'] = val

                if col not in self.feat_dict:
                    self.feat_dict[col] = self.total_cnt
                    self.total_cnt += 1
            # filter empty field
            if e == "":
                continue

            # process categorical fields
            # TODO, here we only keep all values in categorical in dict but not filter
            # feat_dict = {filed_name: {val: count}}
            if col in self.discrete_cols:
                if col in self.feat_dict:
                    if e in self.feat_dict[col]:
                        self.feat_dict[col][e] += 1
                    else:
                        self.feat_dict[col][e] = 1
                else:
                    self.feat_dict[col] = {e: 1}

    def _gen_dict_wrapper(self, chunk_start, chunk_size):
        lock.acquire()  # lock file
        with open(self.filename) as f:
            f.seek(chunk_start)
            lines = f.read(chunk_size).splitlines()
            for id, line in enumerate(lines):
                # process lines
                arr = line.strip('\n').split('\t')
                if len(arr) != 40:
                    # print(id, len(arr))
                    continue
                self._parse(line)
        lock.release()
        return self.feat_dict, self.total_cnt, self.integer_norm

    def build_mp(self):
        """Create values"""
        # TODO, use multi-process
        print("Start to create feature dict...")
        lock = mp.Lock()
        pool = mp.Pool(processes=mp.cpu_count() - 2, initializer=init, initargs=(lock,), maxtasksperchild=1000)

        t_start = time.time()

        jobs = []
        # create jobs
        for chunkStart, chunkSize in chunkify(self.filename):
            jobs.append(pool.apply_async(self._gen_dict_wrapper, (chunkStart, chunkSize)))

        # clean up
        pool.close()
        pool.join()

        t_end = time.time()
        print("Create feature dict complete with multi-thread cost time:{0:.4f}s".format(t_end - t_start))

        # combine all results
        for job in jobs:
            dics, cnt, norm_dics = job.get()
            # parse integer norm dict
            for k, v in norm_dics.items():
                if k in self.integer_norm:
                    if self.integer_norm[k]['min'] < v['min']:
                        self.integer_norm[k]['min'] = v['min']
                    if self.integer_norm[k]['max'] < v['max']:
                        self.integer_norm[k]['max'] = v['max']
                else:
                    self.integer_norm[k] = v

            self.total_cnt = cnt
            for k, v in dics.items():
                if not isinstance(v, dict):
                    if k not in self.feat_dict:
                        self.feat_dict[k] = v
                else:
                    if k in self.feat_dict:
                        for kk, vv in v.items():  # {k: {kk:vv}}
                            if kk in self.feat_dict[k]:
                                self.feat_dict[k][kk] += vv
                            else:
                                self.feat_dict[k][kk] = vv
                    else:
                        self.feat_dict[k] = v
        print("Complete build vocabs: {}, {}".format(len(self.feat_dict), self.total_cnt))

        # values clipped by frequency
        for col in self.discrete_cols:
            # categorical fields was start begin at 14
            clipped_vals = {k: v for k, v in self.feat_dict[col].items() if v > self.clip_values[col - 14]}

            # assign index
            keys = clipped_vals.keys()
            num = len(keys)
            clipped_vals = dict(zip(keys, range(self.total_cnt, self.total_cnt + num)))
            self.total_cnt += num
            self.feat_dict[col] = clipped_vals
        self.feat_dict['<unk>'] = self.total_cnt
        self.total_cnt += 1
        self.feat_dict['feat_size'] = self.total_cnt

        print("Complete build vocabs: {}, {}".format(self.total_cnt, len(self.feat_dict)))
        print("Create integer normalized maximum and minimum value dict: {}".format(len(self.integer_norm)))
        self.save()

    def build(self):
        """Create values"""
        # TODO, use multi-process
        print("Start to create feature dict...")

        t_start = time.time()

        with open(self.filename) as f:
            for line in tqdm(f.readlines()):
                arr = line.strip('\n').split('\t')
                if len(arr) != 40:
                    continue
                self._parse(line)

        t_end = time.time()
        print("Parse line complete with multi-thread cost time:{0:.4f}s".format(t_end - t_start))

        print("Complete build vocabs: {}, {}".format(len(self.feat_dict), self.total_cnt))

        # values clipped by frequency
        t_start = time.time()
        for col in self.discrete_cols:
            # categorical fields was start begin at 14
            clipped_vals = {k: v for k, v in self.feat_dict[col].items() if v > self.clip_values[col - 14]}

            # assign index
            keys = clipped_vals.keys()
            num = len(keys)
            clipped_vals = dict(zip(keys, range(self.total_cnt, self.total_cnt + num)))
            self.total_cnt += num
            self.feat_dict[col] = clipped_vals
        self.feat_dict['<unk>'] = self.total_cnt
        self.total_cnt += 1
        self.feat_dict['feat_size'] = self.total_cnt

        t_end = time.time()
        print("Complete feature filter and vocabs building cost time:{0:.4f}s".format(t_end - t_start))
        print("Total vocab index:{}".format(self.total_cnt))
        self.save()

    def save(self):
        """Save feature dict to file."""
        with open(self.dirs + '/feat_dict.json', 'w') as f:
            json.dump(self.feat_dict, f, indent=4)

        # save integer  normalize dict
        with open(self.dirs + '/integer_norm.json', 'w') as f:
            json.dump(self.integer_norm, f, indent=4)

    def load(self):
        """Load feature dict from disk."""
        t_start = time.time()
        with open(self.dirs + '/feat_dict.json') as f:
            self.feat_dict = json.load(f)
            unk = self.feat_dict['<unk>']
            size = self.feat_dict['feat_size']
            # convert key to column index
            self.feat_dict = {int(k): v for k, v in self.feat_dict.items() if k not in ['<unk>', 'feat_size']}
            self.feat_dict['<unk>'] = unk
            self.feat_dict['feat_size'] = size

        t_end = time.time()

        print("Load feature dict completed total fields:{}, total index: {}".format(len(self.feat_dict),
                                                                                    self.feat_dict['feat_size']))
        print("cost time: {:.4f}s".format(t_end - t_start))

        with open(self.dirs + '/integer_norm.json') as f:
            self.integer_norm = json.load(f)
            self.integer_norm = {int(k): v for k, v in self.integer_norm.items()}
        print("Load integer normalization dict completed total fields: {}".format(len(self.integer_norm)))

    def normalize(self, arr):
        """
        Parse line as input format, we covert raw sample as <raw val \t index \t label or line_num>
        :param arr:
        :return:
        """
        index = []
        values = []
        label = arr[0]

        # process numeric fields
        # TODO, normalize for integer values, store with sparse
        for col in self.numeric_cols:
            if col in self.feat_dict:
                if len(arr[col].strip()) == 0 or arr[col] == '-1' or arr[col] == '0':
                    # values.append(0.0)
                    continue
                else:
                    # TODO, value normalized
                    if int(arr[col]) < self.integer_norm[col]['max']:
                        val = int(arr[col])
                    else:
                        val = self.integer_norm[col]['max']

                    norm = (val - self.integer_norm[col]['min']) / \
                           (self.integer_norm[col]['max'] - self.integer_norm[col]['min'] + 1)

                    norm = "{:.4f}".format(norm).rstrip('0').rstrip('.')
                    if norm == '0':
                        continue

                    values.append(norm)
                    # values.append(arr[col])
                    index.append(self.feat_dict[col])

        # process categorical fields
        for col in self.discrete_cols:
            if col in self.feat_dict:
                if arr[col] in self.feat_dict[col]:

                    if arr[col] == '':
                        # values.append(0.0)
                        continue
                    else:
                        index.append(self.feat_dict[col][arr[col]])
                        values.append(1.0)
                # else:
                #     index.append(self.feat_dict['<unk>'])
                #     values.append(0.0)
        assert len(values) == len(index), "values and index not match"

        return values, index, label

    def _norm_wrapper(self, filename, chunk_start, chunk_size):
        """
        Normalize wrapper
        :param filename: input file name
        :param chunk_start:
        :param chunk_size:
        :return:
        """
        lock.acquire()  # lock file
        with open(filename) as f, open(self.dirs + '/processed.txt', 'a+') as ofs:
            f.seek(chunk_start)
            lines = f.read(chunk_size).splitlines()
            for id, line in enumerate(lines):
                # process lines
                arr = line.strip('\n').split('\t')
                if len(arr) != 40:
                    # print(id, len(arr))
                    continue
                values, index, label = self.normalize(arr)

                values = [str(v) for v in values]
                index = [str(v) for v in index]

                ofs.write(",".join(values) + "\t" + ",".join(index) + "\t" + str(label))
                ofs.write("\n")
        lock.release()

    def dataset_norm(self, filename):
        # TODO, use multi-process
        print("Start to normalize...")
        lock = mp.Lock()
        pool = mp.Pool(mp.cpu_count() - 2, initializer=init, initargs=(lock,))

        t_start = time.time()

        jobs = []
        # create jobs
        for chunkStart, chunkSize in chunkify(filename):
            jobs.append(pool.apply_async(self._norm_wrapper, (filename, chunkStart, chunkSize)))

        # clean up
        pool.close()
        pool.join()

        t_end = time.time()
        print("Normalized completed with multi-thread cost time:{0:.4f}s".format(t_end - t_start))


def main():
    dir = "/root/data/research/data/product/rs/kaggle_DAC/"
    filename = dir + 'test.txt'
    numeric_cols = range(1, 14)
    discrete_cols = range(14, 40)
    clip_values = [20, 5, 200, 200, 5, 1, 50, 5, 1, 90, 40, 200, 50,
                   1, 20, 200, 1, 2, 20, 1, 300, 1, 1, 110, 2, 130]
    ignore_cols = [0]
    genDict = FeatDict(dir, filename, numeric_cols, discrete_cols, ignore_cols, clip_values)
    # genDict.build_mp()
    genDict.load()
    genDict.dataset_norm(filename)


if __name__ == '__main__':
    main()

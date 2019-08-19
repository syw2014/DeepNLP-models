#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : ahocorasick_demo.py
# PythonVersion: python3.5
# Date    : 2019/8/5 13:53
# Software: PyCharm

"""Aho-Corasick multi-pattern string search with pyahocorasick."""

from contextlib import contextmanager
import time
import os
import pickle
import argparse

try:
    import ahocorasick as ahc
    from opencc import OpenCC
except ImportError:
    print("Should install pyahocorasick or opencc-python-reimplemented")


# define parameters
parser = argparse.ArgumentParser()
parser.add_argument("--dict_dir", default=None, help="input dictionary file path")
parser.add_argument("--model_dir", default=None, help="created search model")
parser.add_argument("--input_text", default=None,
                    help="input text file need to search keywords, each string as one line")
parser.add_argument("--output_result", default=None, help="searched results, line with string id and labels")

args = parser.parse_args()


@contextmanager
def timeblock(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print('{} : {}s'.format(label, end - start))


class AHOSearch:
    def __init__(self, dict_dir=None, model_dir=None):
        """
        Create Aho-Corasick Object
        :param dict_dir: dictionary directory, may include more than one files
        :param model_dir: generated aho-corasick automaton model
        """
        self.dict_dir = dict_dir
        self.model_dir = model_dir
        self.ahc = ahc.Automaton()         # aho-corasick trie tree
        self.t2s = OpenCC('t2s')           # convert traditional chinese to simplified chinese

    def create_trie(self):
        """
        Create aho-corasick tree with dictionary
        :return: None
        """
        files = os.listdir(self.dict_dir)
        with timeblock('CreateTrieTree'):
            for filename in files:
                # parse dictionary index
                arr = filename.split('_')
                assert len(arr) == 2, "Require filename as <id_name> but found {}".format(filename)
                label = int(arr[0])
                with open(self.dict_dir + '/' + filename, encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip().lower()
                        tokens = line.split('\t')
                        self.ahc.add_word(tokens[0], (label, tokens[0]))
        print("Create trie tree completed.")

        # convert trie tree to automaton
        with timeblock('CreateAutomaton'):
            self.ahc.make_automaton()

    def save(self):
        """
        Store Aho-Corasick object to file
        :return:
        """
        with open(self.model_dir, 'wb') as f:
            pickle.dump(self.ahc, f)

    def load(self):
        """
        Restore Aho-Corasick object from file
        :return:
        """
        with open(self.model_dir, 'rb') as f:
            self.ahc = pickle.load(f)

    def init(self, reload_dict=False):
        """
        Initialize search object
        :param reload_dict: Bool, whether re-create aho-corasick  trie tree with dictionary
        :return:
        """
        if reload_dict:
            self.create_trie()
        else:
            self.load()

    def search(self, text):
        """
        Search patterns in text
        :param text: input text
        :return: searched pattern and it's label or empty
        """

        # convert to simplified chinese
        text = self.t2s.convert(text.lower())

        results = []
        for k, (idx, tok) in self.ahc.iter(text.strip()):
            results.append([idx, tok])
            # print('Label: {} token:{}'.format(idx, tok))
        return results


def main():
    dict_dir = "D:/github/MiscProjects/data/"
    model_dir = "D:/github/MiscProjects/result/"
    # ahc = AHOSearch(dict_dir, model_dir)
    # ahc.init(reload_dict=True)
    #
    # t1 = "神字幕_狂用葱插我！_哔哩哔哩(゜-゜)つロ干杯~-bilibili"
    # ahc.search(t1)
    #
    # t2 = "神字幕_狂用葱插我！_哔哩哔哩(゜-゜)つロ成人电影干杯~-bilibili"
    # ahc.search(t2)

    assert len(args.dict_dir) > 0, "Input dictionary path {} not exists".format(args.dict_dir)

    # design filter logic
    # create ahc trie tree
    ahc = AHOSearch(args.dict_dir, args.model_dir)
    with timeblock("CreateTrieTree"):
        ahc.init(reload_dict=True)

    # search
    with open(args.input_text, 'r', encoding='utf-8') as f, \
            open(args.output_result, 'w', encoding='utf-8') as fout:
        for line in f.readlines():
            line = line.strip()     # TODO, need more parse
            starts = time.perf_counter()
            res = ahc.search(line)
            ends = time.perf_counter()
            if len(res) == 0:
                res.append([-1, 'None'])
            # duplicate
            strs = ""
            res = dict(res)
            for k, v in res.items():
                strs += str(k) + "," + v + "\t"

            fout.write(line + '\t' + strs.strip() + '\t' + str(len(line)) + "\t" + "{:.6f}".format(ends - starts))
            fout.write('\n')


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author  : Jerry.Shi
# File    : annotation.py
# PythonVersion: python3.6
# Date    : 2022/5/31 14:59
# Software: PyCharm
"""
Label corpus as break or connect with core dictionary and full  dictionary, reference to the paper:
https://arxiv.org/abs/1809.03599
"""

from tqdm import tqdm

# Define same common variables
FILTERED_TYPE = "__FILTERED__"
HALF_WINDOW_SIZE = 7


class TrieNode(object):
    """
    A node in the trie structure
    """

    def __init__(self):
        # Map to store word and it's index
        self.children = dict()
        # Set to store types of each word
        self.types = set()


class Trie(object):
    """
    Trie instances
    """

    def __init__(self):
        # Here the element of trie was TrieNode
        self.trie = list()
        self.stopwords = set()
        # Add one element
        # self.trie.append(TrieNode())

    def get_word_type(self, index):
        """
        Get word type if index was in self.trie list.
        @param index: index of word
        return type if exists, or -1
        """
        if index < 0 or index >= len(self.trie):
            print("Warning: input token/word index {} was not in the trie tree!")
            return -1
        else:
            return self.trie[index].types

    def get_child(self, index, token):
        """
        Get children of token if index was in trie tree
        @param index: the index of token
        @param token: word
        return id of children of token else -1
        """
        if index < 0 or index >= len(self.trie):
            print("Warning: input token/word index {} was not in the trie tree!")
            return -1
        if token in self.trie[index]:
            return self.trie[index][token]
        else:
            return -1

    def is_entity(self, index):
        """
        Check the word of the index whether was an entity or not
        @param index: the index of token
        return true if it's entity else false
        """
        pass

    def insert(self, token_list, type_list, is_lower_case, is_exact_match=False):
        """
        Insert token and it's types in the trie tree
        @param token_list: input token list
        @param type_list: type of token, size was the same as token_list
        @param is_lower_case: whether to insert the lower case into trie tree
        @param is_exact_match: whether to keep the exact match if not,will add Upper case into trie
        """
        # Insert token and type, firstly check the token was already in trie tree if not then insert to trie
        idx = 0
        for token in token_list:
            print("Test, idx, trie-len", idx, len(self.trie))
            if len(self.trie) == 0:
                # insert
                node = TrieNode()
                node.children[token] = idx
                self.trie.append(node)
                # self.trie.append(TrieNode())
            elif token not in self.trie[idx].children:
                self.trie[idx].children[token] = len(self.trie)
                self.trie.append(TrieNode())  # For next process with idx
            idx = self.trie[idx].children[token]
        print("First insert, total: ", len(self.trie))

        # Insert types
        self.trie[idx].types.update(set(type_list))

        # Add all the upper form
        if not is_exact_match:
            idx = 0
            for token in token_list:
                token = token.upper()
                if len(self.trie) == 0:
                    # insert
                    node = TrieNode()
                    node.children[token] = idx
                    self.trie.append(node)
                elif token not in self.trie[idx].children:
                    self.trie[idx].children[token] = len(self.trie)
                    self.trie.append(TrieNode())  # For next process with idx
                idx = self.trie[idx].children[token]
            # Insert types
            self.trie[idx].types.update(set(type_list))

        # Add all the lower form
        if is_lower_case:
            idx = 0
            for token in token_list:
                token = token.lower()
                if len(self.trie) == 0:
                    # insert
                    node = TrieNode()
                    node.children[token] = idx
                    self.trie.append(node)
                elif token not in self.trie[idx].children:
                    self.trie[idx].children[token] = len(self.trie)
                    self.trie.append(TrieNode())  # For next process with idx
                idx = self.trie[idx].children[token]
            # Insert types
            self.trie[idx].types.update(set(type_list))

    def mark_as_filtered(self, tokens_list, no_low_case, is_exact_match=False):
        """
        Insert tokens into trie with FILTERED type, only process full dictionary.
        @param tokens_list: input token list
        @param no_low_case: whether to use low case.
        @param is_exact_match: wether to use exact match
        """
        idx = 0
        for token in tokens_list:
            if token not in self.trie[idx].children:
                self.trie[idx].children[token] = len(self.trie)
                self.trie.append(TrieNode())
            idx = self.trie[idx].children[token]
        if len(self.trie[idx].types) == 0:
            self.trie[idx].types.add(FILTERED_TYPE)

        # Add the all upper form
        if not is_exact_match:
            idx = 0
            for token in tokens_list:
                token = token.upper()
                if token not in self.trie[idx].children:
                    self.trie[idx].children[token] = len(self.trie)
                    self.trie.append(TrieNode())
                idx = self.trie[idx].children[token]
            if len(self.trie[idx].types) == 0:
                self.trie[idx].types.add(FILTERED_TYPE)

        # Add all the lower case
        if not no_low_case:
            idx = 0
            for token in tokens_list:
                token = token.lower()
                if token not in self.trie[idx].children:
                    self.trie[idx].children[token] = len(self.trie)
                    self.trie.append(TrieNode())
                idx = self.trie[idx].children[token]
            if len(self.trie[idx].types) == 0:
                self.trie[idx].types.add(FILTERED_TYPE)

    def remove(self, tokens_list):
        """
        Remove tokens from trie tree
        @param tokens_list: token list

        """
        idx = 0

        for token in tokens_list:
            if token not in self.trie[idx].children:
                # print("Token:{} was not in the trie tree".format(token))
                continue
            idx = self.trie[idx].children[token]
        self.trie[idx].types.clear()

    def token_in_trie(self, tokens_list):
        """
        Check tokens was in trie tree
        @param tokens_list: input token list
        return True if all the token in trie tree else False.
        """
        idx = 0
        for token in tokens_list:
            if token not in self.trie[idx].children:
                return False
            idx = self.trie[idx].children[token]  # get net node idx
        return len(self.trie[idx].types) > 0

    def get_type_from_trie(self, tokens_list):
        """
        Get types of tokens list from trie tree
        """
        idx = 0
        for token in tokens_list:
            if token not in self.trie[idx].children:
                return ""
            idx = self.trie[idx].children[token]

        # Final type
        type_str = ""
        for t in self.trie[idx].types:
            if len(type_str) > 0:
                type_str += ","
            type_str += t
        return type_str

    def clean_stopwords(self, stopwords_file):
        """
        Remove stopword from trie tree
        @param stopwords_file: stopword file
        """
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                self.stopwords.add(line.lower())
                self.remove([line])
                self.remove([line.lower()])
                self.remove([line.upper()])
        print("Clean stopwords total stopwords: {}".format(len(self.stopwords)))


def load_dict(core_dict_file, full_dict_file):
    """
    Load dict from file and create trie tree
    returns Trie Tree instance
    """

    # Create trie tree instance
    trie = Trie()
    # Format of core dictionary:<entity_type_list,\t, keyword>
    # entity_types was split by ','; every line was one keyword
    # eg: Disease \t Vascular Remodeling
    with open(core_dict_file, 'r', encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            arr = line.strip().split('\t')
            if len(arr) != 2:
                continue
            types = arr[0].strip().split(',')
            entity_types = list()
            if len(types) >= 1:
                entity_types.extend(types)

            # TODO, for chinese we remove stopwords for the whole keyword
            # if arr[1] in trie.stopwords or arr[1].lower() in trie.stopwords:
            #     continue
            # Get token list, token was one single word for chinese but in english was one word
            surface_tokens = arr[1].split(' ')
            no_low_case = arr[0].find("PER") != -1 or arr[0].find("OGR") != -1 or arr[0].find("LOC") != -1

            # TODO, for english we remove stopword for single word not the whole keyword
            if not no_low_case:
                for token in surface_tokens:
                    if token in trie.stopwords:
                        no_low_case = True
                        break

            trie.insert(surface_tokens, entity_types, no_low_case)

    print("Core dict inserted!")
    with open(full_dict_file, 'r', encoding='utf-8') as ff:
        # Format of full dictionary was: keyword in each line
        for line in tqdm(ff.readlines()):
            line = line.strip()
            surface_tokens = line.split(' ')
            trie.mark_as_filtered(surface_tokens, no_low_case)

    print("Full dict inserted!")
    return trie


def initialize(core_dict_file, full_dict_file, stopword_file):
    """
    Initialize tire tree
    @param core_dict_file:
    @param full_dict_file:
    @param stopword_file:
    """
    trie = load_dict(core_dict_file, full_dict_file)
    trie.clean_stopwords(stopword_file)
    print("initialized! # of trie nodes =", len(trie.trie))
    return trie


if __name__ == '__main__':
    data_dir = "/home/yw.shi/develop/projects/7.competition/ner-algs/AutoNER-master/data"
    core_dict_file = data_dir + "/BC5CDR/dict_core.txt"
    full_dict_file = data_dir + "/BC5CDR/dict_full.txt"
    stopword_file = data_dir + "/stopwords.txt"

    # Initialize
    initialize(core_dict_file, full_dict_file, stopword_file)


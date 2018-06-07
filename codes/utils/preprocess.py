#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# author: Jerry.Shi
# date: 2018-6-6

"""A data util tools to pre-process data, create vocab from corpus, convert text to id sequence."""

import codecs
from collections import Counter
import tensorflow as tf
import argparse
import jieba

#parser = argparse.ArgumentParser(description="Arugments for data pre-process")
#parser.add_argument("-input_file", type=str, help="input corpus filename")
#parser.add_argument("-min_term_freq", type=int, help="the minmum term frequency threshold")
#parser.add_argument("-output_vocab", type=str, help="output vocabulary filename")

tf.flags.DEFINE_string("input_filename", "../../data/atec_nlp_sim_train.txt", "input corpus filename")
tf.flags.DEFINE_string("stopwords", "../../data/stopwords.txt", "input stopwords filename")
tf.flags.DEFINE_string("output_vocab", "../../data/vocab", "output vocabulary filename")
tf.flags.DEFINE_integer("min_term_freq", 0, "The mimum term frequency threshold")

FLAGS = tf.flags.FLAGS

class Vocab(object):
    """Create vocabulary of word(single word) to id."""
    def __init__(self, filename, is_single_word=False):
        self.filename = filename
        self.is_single_word = is_single_word
        self.vocab = None
        self.stopwrods = None

        # load stop words
        with codecs.open(FLAGS.stopwords, encoding='utf-8') as f:
            words = set()
            for line in f.readlines():
                words.add(line.strip())
            self.stopwords = list(words)
        print("Load Stopwords completed total size: {}".format(len(self.stopwords)))
    
    def _update(counter, word):
        """Insert word into a dict for word count."""
        if counter.has_key(word):
            counter[word] += 1
        else:
            counter[word] = 1

    def _create_vocab(self):
        """Create vocabulary."""
        print("Start creating vocabulary...")
        # To choose single word or token
        #counter = {} or Counter()
        counter = Counter()
        with codecs.open(self.filename, encoding='utf-8') as f:
            for line in f.readlines():
                # TODO, make it as a single character
                # Here 
                arr = line.strip().split("\t")
                if len(arr) < 4:
                    continue
                for w in jieba.cut(arr[1]):
                    if w not in self.stopwords:
                        counter.update(w)
                for w in jieba.cut(arr[2]):
                    if w not in self.stopwords:
                        counter.update(w)
        print("Total words in corpus: {}".format(len(counter)))

        # term filter by term frequency
        word_counts = [w for w in counter.items() if w[1] > FLAGS.min_term_freq]
        word_counts.sort(key=lambda x:x[1], reverse=True)
        vocab = [w[0] for w in word_counts]
        # assign id
        vocab_dict = dict([(x, y) for y, x in enumerate(vocab)])
        # write word count
        with tf.gfile.FastGFile(FLAGS.output_vocab, "w") as f:
            f.write("\n".join(["%s %d" %(w, c) for w, c in vocab_dict.items()]))
        print("Wrote vocabulary to {}".format(FLAGS.output_vocab))
        
        #self.vocab = word_counts

def main(unused_argv):
    vocab = Vocab(FLAGS.input_filename)
    vocab._create_vocab()

if __name__ == "__main__":
    tf.app.run()



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
import time
from tqdm import tqdm


#parser = argparse.ArgumentParser(description="Arugments for data pre-process")
#parser.add_argument("-input_file", type=str, help="input corpus filename")
#parser.add_argument("-min_term_freq", type=int, help="the minmum term frequency threshold")
#parser.add_argument("-output_vocab", type=str, help="output vocabulary filename")

tf.flags.DEFINE_string("input_filename", "../../data/atec_nlp_sim_train.txt", "input corpus filename")
tf.flags.DEFINE_string("stopwords", "../../data/stopwords.txt", "input stopwords filename")
tf.flags.DEFINE_string("output_vocab", "../../data/vocab", "output vocabulary filename")
tf.flags.DEFINE_integer("min_term_freq", 0, "The mimum term frequency threshold")
tf.flags.DEFINE_string("output_filename", "../../data/train.tfrecord", "tfrecord file")

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

        self._create_vocab()
    
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
        t_start = time.time()
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
        t_end = time.time()
        print("Cost time: {}s".format(t_end - t_start))
        
        self.vocab = vocab_dict
        self.vocab['UNK'] = len(vocab_dict)

    def convert_token_to_id(self, token):
        """Convert token to it's id, if it doesn't exist then return the unknown id."""
        if self.vocab.has_key(token):
            return self.vocab[token]
        else:
            return self.vocab['UNK']

    def tokens_to_id_list(self, tokens):
        return list(map(self.convert_token_to_id, tokens))

    def sentence_to_tokens(self, sent):
        """Segment sentence, here you can choose tokenizer."""
        # here we only use single character
        # print(list(sent))
        # Use tokenizer to split sentence
        # tokens = jieba.cut(sent)
        return list(sent)
    
    def sequence_to_id_list(self, sent):
        tokens = self.sentence_to_tokens(sent)
        id_list = self.tokens_to_id_list(tokens)
        return id_list


class BuildTFRecord(Vocab):
    """Build tfrecod."""

    def __init__(self, input_filename, outfile, is_single_word=False):
        super(BuildTFRecord, self).__init__(input_filename, is_single_word)
        self.output_filename = outfile

        self._create_vocab()

    def sequence_to_example(self, query_id_list, doc_id_list, pair_id, label):
        """Convert a pair of text id list to tf example.
        Args:
            query_id_list: The token id list of query
            doc_id_list: The token id list  of doc
            pair_id: The id of this pair in dataset
            label: A integer of 0 or 1 to indicate query and doc are similar
        Returns:
            tf example
        """
        example = tf.train.SequenceExample()

        example.context.feature["pair_id"].int64_list.value.append(pair_id)
        example.context.feature["label"].int64_list.value.append(label)

        # Feature lists
        q_tokens = example.feature_lists.feature_list["q_tokens"]
        d_tokens = example.feature_lists.feature_list["d_tokens"]

        for token in query_id_list:
            q_tokens.feature.add().int64_list.value.append(token)
        for token in doc_id_list:
            d_tokens.feature.add().int64_list.value.append(token)

        return example

    def build_example(self, line):
        """Process one input line to tfrecord."""
        arr = line.split('\t')
        assert len(arr) == 4, "Bad line found in line:{}, {}".format(arr[0], len(arr))
        q_tokens = self.sentence_to_tokens(arr[1])
        query_id_list = self.tokens_to_id_list(q_tokens)
        d_tokens = self.sentence_to_tokens(arr[2])
        doc_id_list = self.tokens_to_id_list(d_tokens)
        pair_id = int(arr[0])
        label = int(arr[-1])

        return self.sequence_to_example(query_id_list, doc_id_list, pair_id, label)

    def convet_corpus_to_tfrecod(self):
        """Convert a whole dataset to tfrecord"""
        print("=={}".format(self.output_filename))
        writer = tf.python_io.TFRecordWriter(self.output_filename)
        with codecs.open(self.filename, encoding="utf-8") as f:
            counter = 0
            for line in tqdm(f.readlines()):
                if len(line.split("\t")) != 4:
                    continue
                example = self.build_example(line)
                if example is not None:
                    writer.write(example.SerializeToString())
                    counter += 1
        print("Wrote data to example finished total example: {}".format(counter))


    @staticmethod
    def example_parse(self, example):
        """Parse example to tf tensor"""
        context_features = {
            "pair_id": tf.FixedLenFeature([], dtype=tf.int64),
            "label": tf.FixedLenFeature([], dtype=tf.int64)
        }

        sequence_features = {
            "q_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "d_tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }

        # parse
        context_parsed, sequence_parsed = tf.parse_single_example(
            serialized=example,
            context_features=context_features,
            sequence_features=sequence_features
        )

        return {"q_tokens": sequence_parsed["q_tokens"], "d_tokens": sequence_parsed["d_tokens"],
                "pair_id": context_parsed["pair_id"], "label": context_parsed["label"]}


def main(unused_argv):
    # vocab = Vocab(FLAGS.input_filename)
    # vocab._create_vocab()
    build_tfr = BuildTFRecord(FLAGS.input_filename, FLAGS.output_filename)
    build_tfr.convet_corpus_to_tfrecod()


if __name__ == "__main__":
    tf.app.run()



#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# File: dssm.py
# Date: 18-6-12 下午3:00

"""A simple implementation of DSSM(Deep Structure Semantic Matchh)"""

import tensorflow as tf

class DSSM(object):
    def __init__(self, parsed_example):
        """Parsed example are include for parts, data id, label, query token id list, doc token id list.
        Args:
            parsed_example: A dict of parsed result, {"pair_id": [batch_size, 1], "label": [batch_size, 1]
                            "q_tokens": [batch_size, None], "d_tokens": [batch_size, None]}
        """

        # train data
        self.q_tokens = parsed_example["q_tokens"]
        self.d_tokens = parsed_example["d_tokens"]
        self.label = parsed_example["label"]
        self.pair_id = parsed_example["pair_id"]


        self.lr = 0.02
        self.dropout_rate = 0.5
        self.hidden_size = [300, 128]
        self.vocab_size = 1640

        with tf.variable_scope("main", initializer=tf.contrib.xavier_initializer):
            # shape = []
            query_embeddings = self.embedding_layer(self.q_tokens)
            doc_embedding = self.embedding_layer(self.d_tokens)


    def embedding_layer(self, input_id):
        """Token id embeddings.
        Args:
            input_id: input token id list
        """
        embeddings = tf.get_variable("embedding", shape=[self.vocab_size, 300])
        text_embeddings = tf.nn.embedding_lookup(params=embeddings, ids= input_id)
        return text_embeddings
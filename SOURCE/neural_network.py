# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""

import tensorflow as tf
import config

class Embedding_Layer():

    def __init__(self, shape):
        self.embedding = tf.Variable(tf.constant(0.0, shape=shape), trainable=False)
        self.embedding_placeholder = tf.placeholder(tf.float32, shape=shape)

    def set_embedding(self, embedding_matrix, session):
        embedding_init = self.embedding.assign(self.embedding_placeholder)
        _ = session.run(embedding_init, feed_dict={self.embedding_placeholder: embedding_matrix})

    def lookup(self, input_data):
        output = tf.nn.embedding_lookup(self.embedding, input_data)
        return output


class RNN_Graph():

    def __init__(self, shape, batch_size):

        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(shape[0],
                                                forget_bias=0.0,
                                                state_is_tuple=True,
                                                reuse=tf.get_variable_scope().reuse)
            cell = tf.contrib.rnn.DropoutWrapper(cell, config.keep_prob)
            return cell

        self.model = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(shape[1])], state_is_tuple=True)
        self.initial_state = self.model.zero_state(batch_size, dtype=tf.float32)

    def feed_forward(self, input_data):
        output, state = tf.nn.dynamic_rnn(self.model, input_data, initial_state=self.initial_state)
        return output, state


class Softmax_Layer():

    def __init__(self, shape):
        self.weights = tf.get_variable("softmax_w", shape=shape, dtype=tf.float32)
        self.biases = tf.get_variable("softmax_b", shape=[shape[1]], dtype=tf.float32)

    def feed_forward(self, input_data):
        logits = tf.nn.softmax(tf.matmul(input_data, self.weights) + self.biases)
        return logits
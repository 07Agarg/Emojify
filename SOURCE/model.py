# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""


import tensorflow as tf
import config
import numpy as np
import neural_network
import os

class MODEL():
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.inputs = tf.placeholder(shape=[None, config.NUM_STEPS, self.vocab_size], dtype=tf.float32)
            self.labels = tf.placeholder(shape=[None, config.NUM_CLASS], dtype=tf.float32)
            self.loss = None
            self.output = None
            self.logits = None
            self.initial_state = None
            self.final_state = None
            
        def build(self, embedding_matrix):
            e_layer = neural_network.Embedding_Layer()
            inputs = e_layer.set_embedding(embedding_matrix)
            rnn_graph = neural_network.RNN_Graph([config.HIDDEN_SIZE, config.NUM_LAYERS])
            output, state = rnn_graph.feed_forward(inputs)
            self.logits = tf.contrib.layers.fully_connected(output[:, -1], activation_fn = tf.nn.softmax)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.logits), axis = 0))
            self.initial_state = rnn_graph.initial_state
            self.final_state = state
            
        def train(self, data):
            optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
            saver = tf.train.Saver()
            with tf.Session as session:
                init = tf.global_variables_initializer()
                session.run(init)
                print("Variables initialized...")
                state = session.run(self.initial_state)
                for epoch in range(config.NUM_EPOCHS):
                    cost = 0
                    total_batch = int(data.size/config.BATCH_SIZE)
                    for i in range(total_batch):
                        batch_X, batch_Y = data.generate_batch()
                        feed_dict = {self.inputs : batch_X, self.output : batch_Y, self.initial_state: state}
                        loss_val, _, state = session.run([self.loss, optimizer, self.final_state], feed_dict = feed_dict)
                        print("batch: ",i , " loss: ", loss_val)
                        cost += (loss_val/total_batch)
                print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(cost))
            saver.save(session, os.path.join(config.MODEL_DIR, "model.ckpt"))
                        
        def test(self, data):
            saver = tf.train.Saver()
            with tf.Session as session:
                saver.restore(session, os.path.join(config.MODEL_DIR, "model.ckpt"))
                state = session.run(self.initial_state)
                for i in range(len(data.dataX)):
                    feed_dict = {self.inputs: [data.dataX[i]], self.initial_state: state}
                    predicted = np.rint(session.run(self.output, feed_dict = feed_dict))
                    print('Actual:', data.dataY[i], 'Predicted:', predicted)
                    
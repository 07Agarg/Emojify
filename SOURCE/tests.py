#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 14:20:15 2018

@author: rahul
"""

import neural_network
import config
import utils
import os
import numpy
import tensorflow as tf
import data

#%%
# =============================================================================
# data.py test
# =============================================================================
word_to_index, index_to_word, word_to_vec, emb_matrix = utils.read_glove_vecs(os.path.join(config.EMBEDDING_DIR, config.EMBEDDING_PATH))
data = data.DATA()
data.read_file(config.TRAIN_PATH, word_to_index)

#%%
# =============================================================================
# Embedding_Layer() test
# =============================================================================
inputs = tf.placeholder(shape=[1,1], dtype=tf.int32)
word_to_index, index_to_word, word_to_vec, emb_matrix = utils.read_glove_vecs(os.path.join(config.EMBEDDING_DIR, config.EMBEDDING_PATH))
e_layer = neural_network.Embedding_Layer(shape=emb_matrix.shape)
with tf.Session() as session:
    init = tf.global_variables_initializer()
    session.run(init)
    e_layer.set_embedding(emb_matrix, session)
    see = session.run(e_layer.embedding)

#%%
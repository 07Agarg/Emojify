# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""


import config
import data
import model
import utils
import os

if __name__ == "__main__":
    # LOAD EMBEDDING
    word_to_index, index_to_word, word_to_vec, emb_matrix = utils.read_glove_vecs(os.path.join(config.EMBEDDING_DIR, config.EMBEDDING_PATH))
    print("Pretrained Embedding Loaded")
    # LOAD DATA
    data = data.DATA()
    data.read_file(config.TRAIN_PATH, word_to_index)
    print("Train data Loaded")
    # BUILD MODEL
    model = model.MODEL(len(word_to_index))
    print("Model Initialized")
    model.build()
    print("Model Built")
    # TRAIN MODEL
    model.train(data, emb_matrix)
    print("Model Trained")
    # TEST MODEL
    data.read_file(config.TEST_PATH, word_to_index)
    print("test data read successfully")
    model.test(data)

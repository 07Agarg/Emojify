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

if __name__ == "main":
    
    word_to_index, index_to_word, word_to_vec, emb_matrix = utils.read_glove_vecs(os.path.join(config.DATA_DIR, config.EMBEDDING_PATH))
    data = data.DATA()
    data.read_file(config.TRAIN_PATH, word_to_index)
    print("train data read successfully")
    
    model = model.Model(len(word_to_index))
    model.build(emb_matrix)
    
    model.train(data)
    
    data.read_file(config.TEST_PATH, word_to_index)
    print("test data read successfully")
    
    model.test(data)
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
import tensorflow as tf

if __name__ == "__main__":
    # LOAD EMBEDDING
    word_to_index, index_to_word, word_to_vec, emb_matrix = utils.read_glove_vecs(os.path.join(config.EMBEDDING_DIR, config.EMBEDDING_PATH))
    print("Pretrained Embedding Loaded")
    
	#LOAD CONFIG
    train_config = config.TrainConfig()
    test_config = config.TestConfig()    
    # LOAD DATA
    train_data = data.DATA(train_config)
    train_data.read_file(config.TRAIN_PATH, word_to_index)
    print("Train data Loaded")
    test_data = data.DATA(test_config)
    test_data.read_file(config.TEST_PATH, word_to_index)
    print("Test data Loaded")
    
    # BUILD MODEL
    #initializer = tf.random_uniform_initializer(train_config.init_scale, train_config.init_scale)												
    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse = None):
            train_model = model.MODEL(train_config, len(word_to_index), training = True)
            train_model.build()
	
    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse = True):
            test_model = model.MODEL(test_config, len(word_to_index), training = False)
            test_model.build()
    print("Model Built")
	
    model_name = os.path.join(config.MODEL_DIR, "model" + str(train_config.BATCH_SIZE) + "_" + str(train_config.NUM_EPOCHS) + ".ckpt")
    #TRAIN MODEL
    train_model.train(train_data, model_name, emb_matrix)
    print("Model Trained")
    # TEST MODEL
    test_model.test(test_data, model_name, emb_matrix)
    print("Model tested")
    

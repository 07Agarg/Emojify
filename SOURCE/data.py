# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""
import config
import os
import pandas as pd
import numpy as np

class DATA():
    
    def __init__(self, filename):
        self.batch_size = config.BATCH_SIZE
        self.batch = None
        self.dataX = None
        self.dataY = None
        self.size = None        
        self.data_index = 0
        
    def sentence_to_indices(self, data, word_to_index):
        sentence_index = np.zeros((self.size, config.NUM_STEPS))
        for i in range(self.size):
            sentence_words = data[i].lower().split()
            for j, word in enumerate(sentence_words):
                sentence_index[i, j] = word_to_index[word]
                
        return sentence_index
            
    def read_file(self, filename, word_to_index):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename), header = None)       
        data = np.asarray(data)
        self.dataX = data[:, :-1]
        self.size = data.size[0]
        self.dataX = self.sentence_to_indices(self.dataX, word_to_index)
        self.dataY = data[:, -1]
        self.dataY = pd.get_dummies(self.dataY)
        
    
    def generate_batch(self):
        batch_X = self.dataX[self.data_index:self.data_index+self.batch_size, :]
        batch_Y = self.dataY[self.data_index:self.data_index+self.batch_size, :]
        self.data_index = (self.data_index + self.batch_size) % self.size
        return batch_X, batch_Y


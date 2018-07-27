# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 20:43:05 2018

@author: ashima.garg

"""
import emoji
import config
import numpy as np 


def label_to_emoji(label):
    return emoji.emojize(config.emoji_dictionary[str(label)], use_aliases = True)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as file:
        word_to_index = {}
        word_to_vec = {}
        index_to_word = {}
        for (i, line) in enumerate(file):
            line = line.strip().split()
            cur_word = line[0]
            word_to_vec[cur_word] = np.array(line[1:], dtype = np.float64)
            word_to_index[cur_word] = i
            index_to_word[i] = cur_word

    vocab_len = len(word_to_index)
    emb_matrix = np.zeros((vocab_len, config.EMBEDDING_SIZE))

    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec[word]

    return word_to_index, index_to_word, word_to_vec, emb_matrix

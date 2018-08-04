# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""
import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
EMBEDDING_DIR = os.path.join(ROOT_DIR, 'EMBEDDING/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_PATH = "train_emoji.csv"
TEST_PATH = "test_emoji.csv"
EMBEDDING_PATH = "glove.6B.50d.txt"

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

emoji_dictionary = {"0": "\u2764\uFE0F",
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

EMBEDDING_SIZE = 50

# MODEL CONFIG
class TrainConfig(object):
	NUM_EPOCHS = 400
	INITIAL_LEARNING_RATE = 0.001
	BATCH_SIZE = 12
	NUM_CLASS = 5
	HIDDEN_SIZE = 256
	NUM_STEPS = 10
	NUM_LAYERS = 2
	keep_prob = 0.5

#TEST CONFIG	
class TestConfig(object):
	NUM_EPOCHS = 400
	INITIAL_LEARNING_RATE = 0.001
	BATCH_SIZE = 1
	NUM_CLASS = 5
	HIDDEN_SIZE = 256
	NUM_STEPS = 10
	NUM_LAYERS = 2
	keep_prob = 0.5
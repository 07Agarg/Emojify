# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:06:16 2018

@author: ashima.garg
"""
import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUT_DIR = os.path.join(ROOT_DIR, 'RESULT/')
MODEL_DIR = os.path.join(ROOT_DIR, 'MODEL/')

TRAIN_PATH = "train_emoji.csv"
TEST_PATH = "test_emoji.csv"
EMBEDDING_PATH = "glove.6B.50d.txt"

# RANDOM NUMBER GENERATOR INFORMATION
SEED = 128

# TRAINING INFORMATION
NUM_EPOCHS = 400
INITIAL_LEARNING_RATE = 0.01
BATCH_SIZE = 5

NUM_CLASS = 5

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

HIDDEN_SIZE = 128
NUM_STEPS = 10
EMBEDDING_SIZE = 50
NUM_LAYERS = 2
keep_prob = 0.5

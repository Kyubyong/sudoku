# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong
'''
import tensorflow as tf
import numpy as np
from hyperparams import Hyperparams as hp
import glob
import re
import os

def load_data(phase="train"):
    '''Loads training / validation data.
    
    Args
      is_train: Boolean. If True, it loads training data.
        Otherwise, it loads validation data.
    
    Returns:
      X: 4-D array of float. Has the shape of (# total games, 9, 9, 1) (for train) 
        or (batch_size, 9, 9, 1) (for validation)
      Y: 3-D array of int. Has the shape of (# total games, 9, 9) (for train) 
        or (batch_size, 9, 9) (for validation)            
    '''
    if phase=="train":
        fpath = hp.train_fpath
    elif phase=="val":
        fpath = hp.val_fpath
    
    lines = open(fpath, 'r').read().splitlines()[1:]
    nsamples = len(lines)
    
    X = np.zeros((nsamples, 9*9), np.float32)  
    Y = np.zeros((nsamples, 9*9), np.int32) 
    
    for i, line in enumerate(lines):
        quiz, solution = line.split(",")
        for j, (q, s) in enumerate(zip(quiz, solution)):
            X[i, j], Y[i, j] = q, s
    
    X = np.reshape(X, (-1, 9, 9))
    Y = np.reshape(Y, (-1, 9, 9))
    return X, Y
        
def get_batch_data():
    '''Returns batch data.
    
    Args:
      is_train: Boolean. If True, it returns batch training data. 
        Otherwise, batch validation data. 
        
    Returns:
      A Tuple of x, y, and num_batch
        x: A `Tensor` of float. Has the shape of (batch_size, 9, 9, 1).
        y: A `Tensor` of int. Has the shape of (batch_size, 9, 9).
        num_batch = A Python int. Number of batches.
    '''
    X, Y = load_data()
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([tf.convert_to_tensor(X), 
                                                  tf.convert_to_tensor(Y)]) 
    
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                  num_threads=8,
                                  batch_size=hp.batch_size, 
                                  capacity=hp.batch_size*64,
                                  min_after_dequeue=hp.batch_size*32, 
                                  allow_smaller_final_batch=False)
    # calc total batch count
    num_batch = len(X) // hp.batch_size 
    
    return x, y, num_batch  # (64, 9, 9, 1), (64, 9, 9), ()

def load_vocab():
    vocab = "E abcdefghijklmnopqrstuvwxyz'" # E: Empty, S: end of Sentence
    char2idx = {char:idx for idx, char in enumerate(vocab)}
    idx2char = {idx:char for idx, char in enumerate(vocab)}
    return char2idx, idx2char    
 
 


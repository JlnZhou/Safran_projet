from skimage.external import tifffile
from skimage import img_as_float32

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import pdb

def read_EM(path):
    x_train = tifffile.imread(path + 'train-volume.tif')
    x_train = img_as_float32(x_train)
    t_train = tifffile.imread(path + 'train-labels.tif')
    t_train = img_as_float32(t_train)
    x_test = tifffile.imread(path + 'test-volume.tif')
    x_test = img_as_float32(x_test)
    return x_train, t_train, x_test

def increase_batch(start, bound, rate=1e-4):
    # increase batch size
    next_size = start
    while True:
        yield math.floor(next_size)
        tmp = next_size*(1+rate)
        if not tmp>bound:
            next_size*=(1+rate)
        
def mini_batch(*x, batch_generator, shuffle=True): 
    # get mini-batch
    ptr = 0
    if isinstance(x, tuple) or isinstance(x, list):
        data_size = x[0].shape[0]
        num_var = len(x) # number of variables
    else:
        data_size = x.shape[0]
    
    x_list = []
    if shuffle:
        order = np.arange(data_size)
        np.random.shuffle(order)
        for _x in x:
            x_list += [_x[order]]
    
    batch_size = batch_generator.__next__()    
    while True:
        if not ptr+batch_size >= data_size:
            yield [_x[ptr:ptr+batch_size] for _x in x_list]
            ptr = ptr + batch_size
            batch_size = batch_generator.__next__()
        else:
            break
    yield [_x[ptr:] for _x in x_list]

def decrease_dropout():
    # decrease dropout rate
    pass

def entropy_loss(x, t):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=x)
    batch_size = loss.get_shape().as_list()[0]
    return tf.reduce_sum(loss) / batch_size

def mean_square_loss(x, t):
    batch_size, input_dim = x.get_shape().as_list()
    if batch_size is None:
        batch_size = 1
    return tf.reduce_sum( tf.square(x - t) ) / (batch_size*input_dim)
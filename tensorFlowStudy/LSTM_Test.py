# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: LSTM_Test
# _date_ = 16/11/28 下午6:40

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#参数
lr=0.001
training_iters=100000
batch_size=128

n_inputs=28
n_steps=28
n_hidden_units=128
n_classes=10

#tf Graph input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])



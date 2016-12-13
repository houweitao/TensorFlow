# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: variable_restore
# _date_ = 16/12/8 上午10:43


import tensorflow as tf
import numpy as np

# restore variable 不能完全重现 network,只是 variable
# redefine teh same shape and same type for your variables

# W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
# b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')


n_hidden_units=128
# my_inputs = 20000 - 44 - 13 + 44 + 13
my_inputs = 1 + 200 + 44 + 1 + 13 + 1

max_steps = 2000

# 如果不是 event,则其他四种没有资格去 judge
event_class = 2 + 1
type_class = 3 + 1
polarity_class = 2 + 1  # TODO
degree_class = 3 + 1
modality_class = 4 + 1

weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([my_inputs, n_hidden_units])),
    # (128, 10)
    # 'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes])),

    # random middle matrix TODO

    'event': tf.Variable(tf.random_normal([n_hidden_units, event_class])),
    'type': tf.Variable(tf.random_normal([n_hidden_units, type_class])),
    'polarity': tf.Variable(tf.random_normal([n_hidden_units, polarity_class])),
    'degree': tf.Variable(tf.random_normal([n_hidden_units, degree_class])),
    'modality': tf.Variable(tf.random_normal([n_hidden_units, modality_class])),
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    # 'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ])),

    'event': tf.Variable(tf.constant(0.1, shape=[event_class, ])),
    'type': tf.Variable(tf.constant(0.1, shape=[type_class, ])),
    'polarity': tf.Variable(tf.constant(0.1, shape=[polarity_class, ])),
    'degree': tf.Variable(tf.constant(0.1, shape=[degree_class, ])),
    'modality': tf.Variable(tf.constant(0.1, shape=[modality_class, ])),
}

# 不用定义 intitial 了

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"save_path/first_save.ckpt")
    print ("weights",sess.run(weights['event']))
    print ("biases", sess.run(biases['degree']))
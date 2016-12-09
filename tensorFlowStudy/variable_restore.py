# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: variable_restore
# _date_ = 16/12/8 上午10:43


import tensorflow as tf
import numpy as np

# restore variable 不能完全重现 network,只是 variable
# redefine teh same shape and same type for your variables

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

# 不用定义 intitial 了

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"save_path/first_save.ckpt")
    print ("weights",sess.run(W))
    print ("biases", sess.run(b))
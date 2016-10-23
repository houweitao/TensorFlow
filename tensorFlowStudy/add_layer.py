# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: add_layer
# _date_ = 16/10/23 下午1:35

# about matrix multply: http://baike.baidu.com/view/2455255.htm

import tensorflow as tf


# activation_function=None : means 为线性函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size]))

    res = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = res
    else:
        outputs = activation_function(res)
    return outputs

# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: CNN
# _date_ = 16/10/23 下午7:30
# https://www.youtube.com/watch?v=pjjH2dGGwwY&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=21
# https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf18_CNN3
# 96%

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)  # 随机变量
    return tf.Variable(inital)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


# x: data
def conv2d(x, W):
    # stride [1,x_move,y_move,1]
    # `"SAME", "VALID"`.
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_poll_2x2(x):
    # 多移动一位: ksize strides
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# placeholder
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print x_image.shape


# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5*5 insize(厚度)=1 outsize(厚度)=32
b_conv1 = bias_variable([32])  # output 32
hidden_layer_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output 28*28*32
hidden_layer_pool1 = max_poll_2x2(hidden_layer_conv1)  # output 因为步长是2,所以是14*14*32

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5*5 insize=32 outsize=64
b_conv2 = bias_variable([64])  # output 64
hidden_layer_conv2 = tf.nn.relu(conv2d(hidden_layer_pool1, W_conv2) + b_conv2)  # output 28*28*64
hidden_layer_pool2 = max_poll_2x2(hidden_layer_conv2)  # output 因为步长是2,所以是7*7*64

# func 1 layer
# full connection
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# If one component of `shape` is the special value -1,
# the size of that dimension is computed so that the total size remains constant.
# In particular, a `shape` of `[-1]` flattens into 1-D.
# At most one component of `shape` can be -1.
h_pool2_flat = tf.reshape(hidden_layer_pool2, [-1, 7 * 7 * 64])  # 打平,3维到1维
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

# func 2 layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# work
sess = tf.Session()
# important step
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))

# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: classification_test
# _date_ = 16/10/23 下午4:15
# https://www.youtube.com/watch?v=aNjdw9w_Qyc&index=17&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import add_layer as layer

# data http://yann.lecun.com/exdb/mnist/
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define placeholder
xs = tf.placeholder(tf.float32, [None, 784])
# xs = tf.placeholder(tf.float(32), [None, 784])  # 28*28个像素点
ys = tf.placeholder(tf.float32, [None, 10])  # 10个输出

# add output layer
predication = layer.add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# another
# b = tf.Variable(tf.zeros([10]))
# W = tf.Variable(tf.zeros([784,10]))
# predication= tf.nn.softmax(tf.matmul(xs,W) + b);

# loss
# neg?
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predication), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())


def compute_accuracy(v_xs, v_ys):
    global predication
    y_pre = sess.run(predication, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# def compute_accuracy1(v_xs, v_ys):
#     global predication
#     y_pre = sess.run(predication, feed_dict={xs: v_xs})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
#     return result


for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次取100
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if step % 50 == 0:
        print compute_accuracy(mnist.test.images, mnist.test.labels)

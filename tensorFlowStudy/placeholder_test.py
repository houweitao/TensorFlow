# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: placeholder_test
# _date_ = 16/10/23 下午1:16

import tensorflow as tf

# input1=tf.placeholder(tf.float32,[2,2])

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)

output=tf.mul(input1,input2)

with tf.Session() as sess:
    # print sess.run(output)
    print sess.run(output,feed_dict={input1:[7.1],input2:[2.]})
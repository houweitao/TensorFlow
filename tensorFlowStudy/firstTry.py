# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: firstTry
# _date_ = 16/10/23 上午2:35

# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np


def generate_data():
    x = np.random.rand(100).astype(np.float32)
    y = x_data * 2 + 0.3
    return x, y


# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 2 + 0.3

# x_data, y_data = generate_data()

print(x_data)
print(y_data)

# print(x_data)

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -3.0, 3.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

        # 2016年10月23日03:10:25

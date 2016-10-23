# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: Visualization_test
# _date_ = 16/10/23 下午3:21
# https://www.youtube.com/watch?v=nhn8B0pM9ls&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=13

# 目前 matplotlib 有 bug,过段时间升级了再看 2016年10月23日15:47:41

import add_layer as layer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 三百个例子,取值-1~1
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)
y_data = np.square(x_data) - 0.5

# None 给多少例子都OK
xs = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.float32)

# 输入层,这里 data 1个属性,so 只有一个神经元
# 隐藏层10个神经元
# 输出层 给出结果,有一个神经元

hidden_layer = layer.add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = layer.add_layer(hidden_layer, 10, 1, activation_function=None)

single_loss = tf.square(ys - prediction)
sum_loss = tf.reduce_sum(single_loss, reduction_indices=[1])
average_loss = tf.reduce_mean(sum_loss)

# important
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(average_loss)

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)
# print x_data

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()
# plt.show(block=False)

for step in range(100):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step % 10 == 0:
        # print step
        # print sess.run(average_loss,feed_dict={xs: x_data, ys: y_data})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass

        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)

        plt.pause(0.1)

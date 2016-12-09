# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: LSTM_morvan
# _date_ = 16/12/1 下午6:18

# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

His code is a very good one for RNN beginners. Feel free to check it out.

https://github.com/MorvanZhou/tutorials/blob/master/tensorflowTUT/tf20_RNN2/full_code.py
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print mnist.train

# hyperparameters
lr = 0.001
training_iters = 100000  # 循环次数
batch_size = 3

n_inputs = 28  # MNIST data input (img shape: 28*28) 每一行有28个元素
n_steps = 28  # time steps 一共是28行
n_hidden_units = 7  # neurons in hidden layer
n_classes = 10  # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.

    # 每一步的 output 都在 outputs 里面. final_state是指(c_state, h_state) 2016年12月04日19:05:23
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs   step * batch * hidden
    # outputs = tf.unpack(tf.transpose(outputs, [0, 1, 2]))  # states is the last outputs     batch * step * hidden
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 所有 batch 的最后结果

    event1 = [tf.matmul(i, weights['out']) + biases['out'] for i in outputs]  # TODO
    print("!!!!!!!")
    print(event1[-1])
    print(len(event1))
    print("!!!!!")

    event1=tf.unpack(tf.transpose(event1,[1,0,2]))

    print("????")
    print(event1[2])
    print(len(event1))
    print("????")



    # outputs = tf.reshape(X_in, [-1, n_steps * batch_size, n_hidden_units])
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 所有 batch 的最后结果

    return results, outputs,event1


pred, outputs,event1 = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# len=tf.unpack(tf.transpose(outputs, [1, 0, 2]))[-2]
# show1=tf.unpack(tf.transpose(outputs, [1, 0, 2]))[-2]
show1 = outputs[2]
# show2=tf.unpack(tf.transpose(outputs, [1, 0, 2]))[-3]
# show3=tf.unpack(tf.transpose(outputs, [1, 0, 2]))[-3]  #batch


init = tf.initialize_all_variables()

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # sess.run(tf.global_variables_initializer())
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # print(batch_ys.shape)
        # break
        #
        # print(batch_ys)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        ret = sess.run([cost], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        print(ret)

        out = sess.run(show1, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        # print(out.shape)
        # print(out)



        event = sess.run(event1, feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        # print(event.shape)
        print(len(event))
        print(len(event[0]))
        print(len(event[0][0]))
        print(event[0])

        break

        if step % 20 == 0:
            # print(sess.run(accuracy, feed_dict={
            #     x: batch_xs,
            #     y: batch_ys,
            # }))
            print ("show....")

            # print(sess.run(show1, feed_dict={
            #     x: batch_xs,
            #     y: batch_ys,
            # }))
            #
            # print(sess.run(show1, feed_dict={
            #     x: batch_xs,
            #     y: batch_ys,
            # }))

            print(sess.run(show3, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))

            step += 1
            break

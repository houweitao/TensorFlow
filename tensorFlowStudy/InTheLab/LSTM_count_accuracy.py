# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: LSTM_count_accuracy
# _date_ = 16/12/13 上午2:29

# TODO 双向LSTM dropout
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import load_data as load
import load_data_embding as load
import datetime
import compare
import numpy as  np
import os
import File_util as util

# import Word

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
# lr = 0.001
lr = 0.02
training_iters = 1000000  # 循环次数
# batch_size = 280
batch_size = 280
# batch_test_size = 140

# n_inputs = 28  # MNIST data input (img shape: 28*28) 每一行有28个元素
# n_steps = 28  # time steps 一共是28行
n_hidden_units = 256  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)

# 单词种类,pos 种类,ne 种类
# my_inputs = 20000 - 44 - 13 + 44 + 13
my_inputs = 1 + 200 + 44 + 1 + 13 + 1
# my_inputs = 1 + 200 + 44 + 1 + 13 + 1

max_steps = 2000

save_tmp = "save_path/@tmp/"
save_path = "save_path/"

# 如果不是 event,则其他四种没有资格去 judge
event_class = 2 + 1
type_class = 3 + 1
polarity_class = 2 + 1  # TODO
degree_class = 3 + 1
modality_class = 4 + 1

x = tf.placeholder(tf.float32, [None, max_steps, my_inputs])  # changable

event_y = tf.placeholder(tf.float32, [None, max_steps, event_class])
type_y = tf.placeholder(tf.float32, [None, max_steps, type_class])
polarity_y = tf.placeholder(tf.float32, [None, max_steps, polarity_class])
degree_y = tf.placeholder(tf.float32, [None, max_steps, degree_class])
modality_y = tf.placeholder(tf.float32, [None, max_steps, modality_class])

# Define weights
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


# def weights_biases_maker(input, output):
#     weight = tf.Variable(tf.random_normal([input, output]))
#     biases = tf.Variable(tf.constant(0.1, shape=[output, ])),


# def my_RNN(X, n, weights, biases):
#     # X: 进来的数据
#     # n: 词语的个数,多少个词语就有多少步
#     pass


# n:段落长度,词语的个数也就是 2016年12月07日20:12:09
def RNN(X):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, my_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, max_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # 每一步的 output 都在 outputs 里面. final_state是指(c_state, h_state) 2016年12月04日19:05:23
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    # 不转可能更好些.一个batch一个batch比较 2016年12月08日12:10:48

    # TODO
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs. step * batch * hidden
    outputs = tf.unpack(tf.transpose(outputs, [0, 1, 2]))  # states is the last outputs. batch * step * hidden
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    # print(weights['event'])

    return outputs


outputs = RNN(x)

eventAll = [tf.matmul(i, weights['event']) + biases['event'] for i in outputs]  # TODO
typeAll = [tf.matmul(i, weights['type']) + biases['type'] for i in outputs]  # TODO
polarityAll = [tf.matmul(i, weights['polarity']) + biases['polarity'] for i in outputs]  # TODO
degreeAll = [tf.matmul(i, weights['degree']) + biases['degree'] for i in outputs]  # TODO
modalityAll = [tf.matmul(i, weights['modality']) + biases['modality'] for i in outputs]  # TODO

cost = tf.reduce_mean(
    (event_class - 1) * tf.nn.softmax_cross_entropy_with_logits(eventAll, event_y)
    + (type_class + 1) * tf.nn.softmax_cross_entropy_with_logits(typeAll, type_y)
    + (polarity_class + 1) * tf.nn.softmax_cross_entropy_with_logits(polarityAll, polarity_y)
    + (degree_class - 1) * tf.nn.softmax_cross_entropy_with_logits(degreeAll, degree_y)
    + (modality_class + 2) * tf.nn.softmax_cross_entropy_with_logits(modalityAll, modality_y))


# cost = tf.reduce_mean(
#     8 * tf.nn.softmax_cross_entropy_with_logits(eventAll, event_y)
#     + tf.nn.softmax_cross_entropy_with_logits(typeAll, type_y)
#     + tf.nn.softmax_cross_entropy_with_logits(polarityAll, polarity_y)
#     + tf.nn.softmax_cross_entropy_with_logits(degreeAll, degree_y)
#     + tf.nn.softmax_cross_entropy_with_logits(modalityAll, modality_y))


def show_cost(cost):
    # print(str(cost))
    return tf.train.AdamOptimizer(lr).minimize(cost)


train_op = show_cost(cost)
# tf.train.AdamOptimizer(lr).minimize(cost)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # sess.run(tf.global_variables_initializer())
    sess.run(init)

    count = 0
    # count = 357
    # count = 714
    # pre = "save_path/"
    # after = "/LSTM.ckpt"
    # cur_path = pre + str(count) + after
    # saver.restore(sess, cur_path)

    step = count
    # step = 0
    time_style = "%Y-%m-%d %H:%M:%S"

    batch_word, batch_event, batch_type, batch_polarity, batch_degree, batch_modality = load.get_train_batches(
        batch_size)
    batch_xs = batch_word.reshape([batch_size, max_steps, my_inputs])

    while (step - count) * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape([batch_size, max_steps, my_inputs])

        now = datetime.datetime.now()
        print(step, "begin load data:", now.strftime(time_style))

        sess.run([train_op], feed_dict={
            x: batch_xs,
            event_y: batch_event,
            type_y: batch_type,
            polarity_y: batch_polarity,
            degree_y: batch_degree,
            modality_y: batch_modality,
            # keep_prob: 0.8
        })

        cost_ret = ""
        if step % 1 == 0:
            cost_ret = sess.run(cost, feed_dict={
                x: batch_xs,
                # weights:weights,
                # biases:biases,
                event_y: batch_event,
                type_y: batch_type,
                polarity_y: batch_polarity,
                degree_y: batch_degree,
                modality_y: batch_modality,
                # keep_prob: 1
            })
            print("cost: ", cost_ret)

        if step % 1 == 0:
            # TODO 用一个对象把这些东西都包起来

            ret = sess.run(outputs, feed_dict={
                x: batch_xs,
                # keep_prob: 1
            })
            ret = load.normalize_form(ret)

            # print(len(ret))
            # print(ret[0])
            # print("output!! ", ret.shape)
            # print(ret.shape)

            # print("---")

            weights_change = sess.run(weights)
            biases_change = sess.run(biases)

            # weights=load.normalize_form(weights)
            # biases=load.normalize_form(biases)

            # print (weights_change['event'].shape)

            eventAll = [np.dot(i, np.matrix(weights_change['event'])) + np.matrix(biases_change['event']) for i in
                        ret]  # TODO
            typeAll = [np.dot(i, np.matrix(weights_change['type'])) + np.matrix(biases_change['type']) for i in
                       ret]  # TODO
            polarityAll = [np.dot(i, np.matrix(weights_change['polarity'])) + np.matrix(biases_change['polarity']) for i
                           in
                           ret]  # TODO
            degreeAll = [np.dot(i, np.matrix(weights_change['degree'])) + np.matrix(biases_change['degree']) for i in
                         ret]  # TODO
            modalityAll = [np.dot(i, np.matrix(weights_change['modality'])) + np.matrix(biases_change['modality']) for i
                           in
                           ret]  # TODO

            eventAll = load.normalize_form(eventAll)
            typeAll = load.normalize_form(typeAll)
            polarityAll = load.normalize_form(polarityAll)
            degreeAll = load.normalize_form(degreeAll)
            modalityAll = load.normalize_form(modalityAll)

            # print(eventAll.shape)
            # print(batch_event.shape)

            e, t, p, d, m = compare.compare_five_p_and_r(eventAll, batch_event, typeAll, batch_type, polarityAll,
                                                         batch_polarity, degreeAll,
                                                         batch_degree,
                                                         modalityAll, batch_modality)

            # e, t, p, d, m = compare.compare_five(eventAll, batch_event, typeAll, batch_type, polarityAll,
            #                                  batch_polarity, degreeAll,
            #                                  batch_degree,
            #                                  modalityAll, batch_modality)

        print(e, t, p, d, m)

        f = open('save_path/record.txt', 'a')
        f.write('step ' + str(step) + '\n')
        f.write('cost ' + str(cost_ret))
        f.write('\n')
        line = str(e) + ';' + str(t) + ';' + str(p) + ';' + str(d) + ';' + str(m)
        f.write('precision ' + line)
        f.write('\n')
        f.write('\n')
        f.close()

        tmp_dir = save_tmp + str(step)
        os.mkdir(tmp_dir)
        path = tmp_dir + "/LSTM.ckpt"
        saver.save(sess, path)
        # print(sp)

        save_dir = save_path + str(step)
        util.copy(tmp_dir, save_dir)

        step += 1

        now = datetime.datetime.now()
        print(step, "end load data:", now.strftime(time_style))

# if step == 30:
#     break

# print(sess.run(weights))
# save_path = saver.save(sess, "save_path/LSTM.ckpt")
# print ('Save to path', save_path)

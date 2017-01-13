# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: LSTM_resotre
# _date_ = 16/12/13 下午2:56

# TODO 双向LSTM dropout
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import load_data as load
import load_data_embding as load
import datetime
import compare
import numpy as  np
import Word

# set random seed for comparing the two result calculations
# tf.set_random_seed(1)

# this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
batch_size = 140

# n_inputs = 28  # MNIST data input (img shape: 28*28) 每一行有28个元素
# n_steps = 28  # time steps 一共是28行
n_hidden_units = 128  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)

# 单词种类,pos 种类,ne 种类
# my_inputs = 20000 - 44 - 13 + 44 + 13
my_inputs = 1 + 200 + 44 + 1 + 13 + 1

max_steps = 2000

# 如果不是 event,则其他四种没有资格去 judge
event_class = 2 + 1
type_class = 3 + 1
polarity_class = 2 + 1  # TODO
degree_class = 3 + 1
modality_class = 4 + 1

# tf Graph input
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


# n:段落长度,词语的个数也就是 2016年12月07日20:12:09
def RNN(X):
    X = tf.reshape(X, [-1, my_inputs])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, max_steps, n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unpack(tf.transpose(outputs, [0, 1, 2]))  # states is the last outputs. batch * step * hidden
    return outputs


outputs = RNN(x)

eventAll = [tf.matmul(i, weights['event']) + biases['event'] for i in outputs]  # TODO
typeAll = [tf.matmul(i, weights['type']) + biases['type'] for i in outputs]  # TODO
polarityAll = [tf.matmul(i, weights['polarity']) + biases['polarity'] for i in outputs]  # TODO
degreeAll = [tf.matmul(i, weights['degree']) + biases['degree'] for i in outputs]  # TODO
modalityAll = [tf.matmul(i, weights['modality']) + biases['modality'] for i in outputs]  # TODO

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(eventAll, event_y)
                      + tf.nn.softmax_cross_entropy_with_logits(typeAll, type_y)
                      + tf.nn.softmax_cross_entropy_with_logits(polarityAll, polarity_y)
                      + tf.nn.softmax_cross_entropy_with_logits(degreeAll, degree_y)
                      + tf.nn.softmax_cross_entropy_with_logits(modalityAll, modality_y))

# train_op = tf.train.AdamOptimizer(lr).minimize(cost)

saver = tf.train.Saver()


def run_once(ckpt_path):
    with tf.Session() as sess:
        # saver.restore(sess, "save_path/10.bak/LSTM.ckpt")
        saver.restore(sess, ckpt_path)
        # sess.run(init)
        time_style = "%Y-%m-%d %H:%M:%S"

        batch_word, batch_event, batch_type, batch_polarity, batch_degree, batch_modality = load.get_test_batches(
            batch_size)
        batch_xs = batch_word.reshape([batch_size, max_steps, my_inputs])

        ret = sess.run(outputs, feed_dict={
            x: batch_xs,
        })
        ret = load.normalize_form(ret)

        weights_change = sess.run(weights)
        biases_change = sess.run(biases)

        eventAll = [np.dot(i, np.matrix(weights_change['event'])) + np.matrix(biases_change['event']) for i in
                    ret]
        typeAll = [np.dot(i, np.matrix(weights_change['type'])) + np.matrix(biases_change['type']) for i in
                   ret]
        polarityAll = [np.dot(i, np.matrix(weights_change['polarity'])) + np.matrix(biases_change['polarity']) for i
                       in
                       ret]
        degreeAll = [np.dot(i, np.matrix(weights_change['degree'])) + np.matrix(biases_change['degree']) for i in
                     ret]
        modalityAll = [np.dot(i, np.matrix(weights_change['modality'])) + np.matrix(biases_change['modality']) for i
                       in
                       ret]

        eventAll = load.normalize_form(eventAll)
        typeAll = load.normalize_form(typeAll)
        polarityAll = load.normalize_form(polarityAll)
        degreeAll = load.normalize_form(degreeAll)
        modalityAll = load.normalize_form(modalityAll)

        print(
            compare.compare_five(eventAll, batch_event, typeAll, batch_type, polarityAll, batch_polarity, degreeAll,
                                 batch_degree,
                                 modalityAll, batch_modality))


def run_re(low, high):
    with tf.Session() as sess:
        batch_word, batch_event, batch_type, batch_polarity, batch_degree, batch_modality = load.get_test_batches(
            batch_size)
        batch_xs = batch_word.reshape([batch_size, max_steps, my_inputs])

        count = low
        pre = "save_path/"
        after = "/LSTM.ckpt"
        while count <= high:
            cur_path = pre + str(count) + after

            saver.restore(sess, cur_path)
            # sess.run(init)
            time_style = "%Y-%m-%d %H:%M:%S"

            ret = sess.run(outputs, feed_dict={
                x: batch_xs,
            })
            ret = load.normalize_form(ret)

            weights_change = sess.run(weights)
            biases_change = sess.run(biases)

            eventAll = [np.dot(i, np.matrix(weights_change['event'])) + np.matrix(biases_change['event']) for i in
                        ret]
            typeAll = [np.dot(i, np.matrix(weights_change['type'])) + np.matrix(biases_change['type']) for i in
                       ret]
            polarityAll = [np.dot(i, np.matrix(weights_change['polarity'])) + np.matrix(biases_change['polarity']) for i
                           in
                           ret]
            degreeAll = [np.dot(i, np.matrix(weights_change['degree'])) + np.matrix(biases_change['degree']) for i in
                         ret]
            modalityAll = [np.dot(i, np.matrix(weights_change['modality'])) + np.matrix(biases_change['modality']) for i
                           in
                           ret]

            eventAll = load.normalize_form(eventAll)
            typeAll = load.normalize_form(typeAll)
            polarityAll = load.normalize_form(polarityAll)
            degreeAll = load.normalize_form(degreeAll)
            modalityAll = load.normalize_form(modalityAll)

            e, t, p, d, m = compare.compare_five_p_and_r(eventAll, batch_event, typeAll, batch_type, polarityAll,
                                                         batch_polarity, degreeAll,
                                                         batch_degree,
                                                         modalityAll, batch_modality)

            print(count, e, t, p, d, m)

            f = open('save_path/restore.txt', 'a')
            f.write('step ' + str(count) + '\n')
            line = str(e) + ';' + str(t) + ';' + str(p) + ';' + str(d) + ';' + str(m)
            f.write('precision ' + line)
            f.write('\n')
            f.write('\n')
            f.close()

            count += 10


# run_re(72, 357)
# run_re(989, 990)
run_re(0, 761)
# run_once("save_path/10.bak/LSTM.ckpt")

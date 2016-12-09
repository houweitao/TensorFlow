# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: LSTM
# _date_ = 16/12/8 下午4:54


# TODO 双向LSTM dropout
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import load_data as load
import numpy as np

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000  # 循环次数
batch_size = 1

# n_inputs = 28  # MNIST data input (img shape: 28*28) 每一行有28个元素
# n_steps = 28  # time steps 一共是28行
n_hidden_units = 128  # neurons in hidden layer
# n_classes = 10  # MNIST classes (0-9 digits)

# 单词种类,pos 种类,ne 种类
my_inputs = 20000 - 44 - 13 + 44 + 13

max_steps = 5000

# 如果不是 event,则其他四种没有资格去 judge
event_class = 2 + 1
type_class = 3 + 1
polarity_class = 2 + 1  # TODO
degree_class = 3 + 1
modality_class = 4 + 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, max_steps, my_inputs])  # changable

# y = tf.placeholder(tf.float32, [None, n_classes])
# step_len=tf.placeholder(tf.float32, [None, n_steps, my_inputs])


# def makeX(step):
# tf.placeholder(tf.float32, [None, step, my_inputs])


# TODO
# event_y = tf.placeholder(tf.float32, [None, event_class * max_steps])
# type_y = tf.placeholder(tf.float32, [None, type_class * max_steps])
# polarity_y = tf.placeholder(tf.float32, [None, polarity_class * max_steps])
# degree_y = tf.placeholder(tf.float32, [None, degree_class * max_steps])
# modality_y = tf.placeholder(tf.float32, [None, modality_class * max_steps])


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

saver = tf.train.Saver()


# def weights_biases_maker(input, output):
#     weight = tf.Variable(tf.random_normal([input, output]))
#     biases = tf.Variable(tf.constant(0.1, shape=[output, ])),


# def my_RNN(X, n, weights, biases):
#     # X: 进来的数据
#     # n: 词语的个数,多少个词语就有多少步
#     pass


# n:段落长度,词语的个数也就是 2016年12月07日20:12:09
def RNN(X, weights, biases):
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
    # 不转可能更好些.一个batch一个batch比较 2016年12月08日12:10:48

    # TODO
    # outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs. step * batch * hidden
    outputs = tf.unpack(tf.transpose(outputs, [0, 1, 2]))  # states is the last outputs. batch * step * hidden
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    # outputs = tf.reshape(outputs, [-1, , n_hidden_units])
    eventAll = [tf.matmul(i, weights['event']) + biases['event'] for i in outputs]  # TODO
    typeAll = [tf.matmul(i, weights['type']) + biases['type'] for i in outputs]  # TODO
    polarityAll = [tf.matmul(i, weights['polarity']) + biases['polarity'] for i in outputs]  # TODO
    degreeAll = [tf.matmul(i, weights['degree']) + biases['degree'] for i in outputs]  # TODO
    modalityAll = [tf.matmul(i, weights['modality']) + biases['modality'] for i in outputs]  # TODO

    # list->array
    # eventAll = np.array(eventAll)
    # typeAll = np.array(typeAll)
    # polarityAll = np.array(polarityAll)
    # degreeAll = np.array(degreeAll)
    # modalityAll = np.array(modalityAll)

    # print(eventAll)

    # event = tf.matmul(outputs[-1], weights['event']) + biases['event']
    # type = tf.matmul(outputs[-1], weights['type']) + biases['type']
    # polarity = tf.matmul(outputs[-1], weights['polarity']) + biases['polarity']
    # degree = tf.matmul(outputs[-1], weights['degree']) + biases['degree']
    # modality = tf.matmul(outputs[-1], weights['modality']) + biases['modality']

    return eventAll, typeAll, polarityAll, degreeAll, modalityAll


event, type, polarity, degree, modality = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(event, event_y)
+ tf.nn.softmax_cross_entropy_with_logits(type, type_y)
+ tf.nn.softmax_cross_entropy_with_logits(polarity, polarity_y)
+ tf.nn.softmax_cross_entropy_with_logits(degree, degree_y)
+ tf.nn.softmax_cross_entropy_with_logits(modality, modality_y))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_event = tf.equal(tf.argmax(event, 1), tf.argmax(event_y, 1))
correct_type = tf.equal(tf.argmax(type, 1), tf.argmax(type_y, 1))
correct_polarity = tf.equal(tf.argmax(polarity, 1), tf.argmax(polarity_y, 1))
correct_degree = tf.equal(tf.argmax(degree, 1), tf.argmax(degree_y, 1))
correct_modality = tf.equal(tf.argmax(modality, 1), tf.argmax(modality_y, 1))

# TODO
accuracy_event = tf.reduce_mean(tf.cast(correct_event, tf.float32))
accuracy_type = tf.reduce_mean(tf.cast(correct_type, tf.float32))
accuracy_polarity = tf.reduce_mean(tf.cast(correct_polarity, tf.float32))
accuracy_degree = tf.reduce_mean(tf.cast(correct_degree, tf.float32))
accuracy_modality = tf.reduce_mean(tf.cast(correct_modality, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # sess.run(tf.global_variables_initializer())
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # batch_xs = batch_xs.reshape([batch_size, max_steps, my_inputs])

        batch_word, batch_event, batch_type, batch_polarity, batch_degree, batch_modality = load.get_batches(batch_size)
        batch_xs = batch_word.reshape([batch_size, max_steps, my_inputs])

        sess.run([train_op], feed_dict={
            x: batch_xs,
            event_y: batch_event,
            type_y: batch_type,
            polarity_y: batch_polarity,
            degree_y: batch_degree,
            modality_y: batch_modality
        })
        if step % 1 == 0:
            print(sess.run(cost, feed_dict={
                x: batch_xs,
                event_y: batch_event,
                type_y: batch_type,
                polarity_y: batch_polarity,
                degree_y: batch_degree,
                modality_y: batch_modality
            }))

        # if step % 1 == 0:
        #     ev = sess.run(event, feed_dict={
        #         x: batch_xs,
        #         event_y: batch_event,
        #         type_y: batch_type,
        #         polarity_y: batch_polarity,
        #         degree_y: batch_degree,
        #         modality_y: batch_modality
        #     })
        #
        #     print(ev.shape)
        #     print(ev[0].shape)
        #     print(ev[0][0].shape)
        #
        #     print(len(ev))
        #     print(len(ev[0]))
        #     print(len(ev[0][0]))
        #     print(ev)

        # show_output()
        step += 1
    save_path = saver.save(sess, "save_path/LSTM.ckpt")
    print ('Save to path', save_path)

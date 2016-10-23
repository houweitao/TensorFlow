# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: variable_test
# _date_ = 16/10/23 下午1:09

import tensorflow as tf

state = tf.Variable(0, name='counter')
# print state.name

one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 如果定义了 variable 一定要有这个
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(3):
    sess.run(update)
    print sess.run(state)
sess.close()

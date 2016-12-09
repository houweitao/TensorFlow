# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: variable_saver
# _date_ = 16/12/8 上午10:43

import tensorflow as tf

# Save to file
W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.initialize_all_variables()

saver =tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,"save_path/first_save.ckpt")
    print ('Save to path',save_path)

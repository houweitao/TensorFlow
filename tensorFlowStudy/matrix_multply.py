# -*- coding: utf-8 -*-
# _author_ = 'hou'
# _project_: matrix_multply
# _date_ = 16/10/23 下午1:00

import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [1]])

product = tf.matmul(matrix1, matrix2)

# product2=np.dot(matrix1,matrix2)

print "hah"
print product
# print product2

# method 1

sess = tf.Session()
result = sess.run(product)
print result
sess.close()

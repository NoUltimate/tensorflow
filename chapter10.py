# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:38:02 2018

@author: 永青一页
"""
import tensorflow as tf

m1 = tf.constant([[2, 2]])
m2 = tf.constant([[3],
                  [3]])
dot_operation = tf.matmul(m1, m2)

print(dot_operation)  # wrong! no result

# method1 use session
sess = tf.Session()
result = sess.run(dot_operation)
print(result)
sess.close()


#method2
with tf.Session() as sess:  #sess只存在于with语句块内，出去后自动结束
    result2=sess.run(dot_operation)
    print(result2)
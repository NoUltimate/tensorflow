# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 22:50:50 2018

@author: NoUltimate
"""

import tensorflow as tf
import numpy as np

## Save to file
##remember to define th same dtype and shape when restore
# W=tf.Variable([1,2,3],[3,4,5],dtype=tf.float32,name='weights')
# b=tf.Variable([1,2,3],dtype=tf.float32,name='biases')


#  init=tf.global_variables_initializer()


# saver=tf.train.Saver()


# with tf.Session() as sess:
# 	sess.run(init)
# 	save_path=saver.save(sess,"my_net/save_net.ckpt")
# 	print("Save to path:",save_path)


#restore variables
#redefine the same shape and same type for your variables
#arange 创造等差数组

W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

#not need init step

tf.reset_default_graph()
restore_graph = tf.Graph()
with tf.Session(graph=restore_graph) as sess:
    restore_saver = tf.train.import_meta_graph('my_net/save_net.ckpt.meta')
    restore_saver.restore(sess,tf.train.latest_checkpoint('my_net/'))
    print("weights",sess.run(W))
    print("biases",sess.run(b))
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:03:42 2018

@author: 永青一页
"""

import tensorflow as tf

input1=tf.placeholder(tf.float32)#在run的时候从外界传入数值
input2=tf.placeholder(tf.float32)


output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.], input2:[2.]}))
    #词典
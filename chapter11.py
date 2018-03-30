# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:54:38 2018

@author: 永青一页
"""

import tensorflow as tf

state=tf.Variable(0,name='counter')#定义他是变量QAQ
#print(state.name)
one=tf.constant(1)

new_value=tf.add(state,one)
update=tf.assign(state,new_value)#把state的值变为new_value


init=tf.initialize_all_variables()#初始化所有变量

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
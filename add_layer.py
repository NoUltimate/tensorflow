# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:44:42 2018

@author: 永青一页
"""

import tensorflow as tf
import numpy as np
#加入神经层
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]#np.newaxis加一个维度，300个例子
noise=np.random.normal(0,0.05,x_data.shape)#min=0，方差0.05，格式和x_data一样
y_data=np.square(x_data)-0.5+noise


xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])
#建造第一层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)


loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#reduction_indices 按行求和 mean 平均值
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#以0.1的效率的步差进行更正

init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50==0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
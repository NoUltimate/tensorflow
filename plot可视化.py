# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 21:06:17 2018

@author: 永青一页
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt#可视化

#加入神经层
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]#np.newaxis加一个维度，300个例子
noise=np.random.normal(0,0.05,x_data.shape)#min=0，方差0.05，格式和x_data一样
y_data=np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):   
     xs=tf.placeholder(tf.float32,[None,1],name='x_input')
     ys=tf.placeholder(tf.float32,[None,1],name='y_input')
#建造第一层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)

with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
#reduction_indices 按行求和 mean 平均值
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#以0.1的效率的步差进行更正

init=tf.initialize_all_variables()
sess=tf.Session()
writer=tf.summary.FileWriter('logs/',sess.graph)
sess.run(tf.initialize_all_variables())


fig=plt.figure()#图片框
ax=fig.add_subplot(1,1,1)#图像应该是1*1的，且当前选中的是1个subplot中的第一个
ax.scatter(x_data,y_data)#点
plt.ion()#让show函数执行后不会使程序终止
plt.show()

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i% 50==0:
       try:
          ax.lines.remove(lines[0])
       except Exception:
           pass           
       print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
       prediction_value=sess.run(prediction,feed_dict={xs:x_data})
       line=ax.plot(x_data,prediction_value,'r-',lw=5)#plot:曲线 x轴数据，y轴数据，红色，线的宽度
       plt.pause(0.1)

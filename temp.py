# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

### creater tensorflow structure start###
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))##一维，数的范围在-1到1
biases=tf.Variable(tf.zeros([1]))


y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))##平均值
optimizer=tf.train.GradientDescentOptimizer(0.5)##梯度下降优化
train=optimizer.minimize(loss)


init=tf.initialize_all_variables()#初始化
### creater tensorflow structure start###


sess=tf.Session()##sess指向要处理的地方
sess.run(init)##激活


for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))##Weights是张量无法直接输出





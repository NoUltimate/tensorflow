
# coding: utf-8

# In[4]:

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

##初始输入和输出数据，784个像素的图片
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])


##正向传播
#权重和偏重定义
w = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([1,10]) + 0.1,)
#预测值
prediction=tf.nn.softmax(tf.matmul(xs,w)+b)


##反向传播
#定义cross_entropy 信息熵 判断模型对真实概率分布估计的准确程度
#这里cross_entropy就是损失函数  reduction_indices 不同方向压缩维度
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))  

#学习速率 优化目标
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


##初始化
sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000): 
    batch_xs,batch_ys=mnist.train.next_batch(100)
        #将mnist的数据给x和y_用来训练
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        global prediction
        #计算训练结果的准确度，argmax是返回最大的序列
        correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
        #equal返回的是bool类型，这里要转换(cast：类型转换函数)并且求平均
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        result = sess.run(accuracy, feed_dict={ys:mnist.test.labels})
        print(result)
            


# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 13:18:24 2018

@author: 永青一页
"""

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
#lord data
digits = load_digits()
#训练集
X = digits.data#手写数字特征向量数据集，每一个元素都是一个64维的特征向量。
#标记
y = digits.target#特征向量对应的标记，每一个元素都是自然是0-9的数字。
#对标记进行二值化
y = LabelBinarizer().fit_transform(y)
#切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
#加入神经层
def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs
 
keep_prob = tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])
#建造第一层
l1=add_layer(xs,64,100,'l1',activation_function=tf.nn.tanh)
prediction=add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#以0.6的效率的步差进行更正

sess=tf.Session()
merged=tf.summary.merge_all()
train_writer=tf.summary.FileWriter("logs/train",sess.graph)
test_writer=tf.summary.FileWriter("logs/test",sess.graph)
sess.run(tf.global_variables_initializer())


for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob: 0.5})
    if i%50==0:
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer=train_writer.add_summary(train_result,i)
        test_writer=test_writer.add_summary(test_result,i)

# coding: utf-8

# In[22]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义权重和偏重
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
##初始化时应该加入少量的噪声来打破对称性以及避免0梯度。由于
##我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    #tf.cast-类型转换
    ## tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素,如果是相等的那就返回True,反正返回False,返回的值的矩阵维度和A是一样的
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    #函数y=f(x)，x0= argmax(f(x)) 的意思就是参数x0满足f(x0)为f(x)的最大值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    #keep_prob是保留概率，即我们要保留的RELU的结果所占比例
    return result


# In[14]:


#定义卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    


# In[16]:


#定义xs,ys
xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1])

#正向传播
##conv1
##32个筛选器，大小为5x5
w_conv1=weight_variable([5,5,1,32])  
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)  #14x14x32

##conv1
##32个筛选器，大小为5x5
w_conv2=weight_variable([5,5,32,64])  
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)  #7x7x64



# In[17]:


#function layer1
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)


# In[18]:


#function layer2
w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
#softmax的作用--输出是10个
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)


# In[19]:


#反向传播
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# In[25]:


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50==0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))


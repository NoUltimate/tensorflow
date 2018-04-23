
# coding: utf-8

# In[26]:


import tensorflow as tf

def inference(images,batch_size,n_classes):
    #卷积层1
    #16个3x3的卷积核（3通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    #tf.bariable_scope:实现 变量共享
    with tf.variable_scope("conv1") as scope:
        #生成截断正态分布的随机数
        #16个3x3的卷积核（3通道）
        weights=tf.get_variable('weights',shape=[3,3,3,16],dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases=tf.get_variable('biases',shape=[16],ftype=tf.float32,initializer=tf.constant_initializer(0.1))
        #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
        conv=tf.nn.conv2d(images,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(pre_activation,name='conv1')

    #pool层1
    #3x3最大池化，步长strides为2，池化后执行lrn()操作，局部响应归一化，对训练有利。
    with tf.variable_scope('pooling1_lrn') as scope:
        #第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
        pool1=tf.nn.max_pool(conv1,ksize=[1,1,1,1],strides=[1,1,1,1],padding='SAME',name='pooling1')
        #归一化操作
        norm1=tf.nn.lrn(pool1,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')
    
    #卷积层2
    #16个3x3的卷积核（16通道），padding=’SAME’，表示padding后卷积的图与原图尺寸一致，激活函数relu()
    with tf.variable_scope('conv2',reuse=True) as scope:
        weights=tf.get_variable('weights',shape=[3,3,16,16],dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases=tf.get_variable('biases',shape=[16],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(norm1,weights,strides=[1,1,1,1],padding='SAME')
        pre_activation=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(pre_activation,name='conv2')
        
    #pool层2
    #3x3最大池化，步长strides为2，池化后执行lrn()操作
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2=tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,1,1,1],padding='SAME',name='pooling2')
        norm2=tf.nn.lrn(pool2,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')
        
        #全连接层3
    #128个神经元，将之前pool层的输出reshape成一列，激活函数relu()
    with tf.variable_scope('local3') as scope:
        reshape=tf.reshape(pool2,shape=[batch_size,-1])
        dim=reshape.get_shape()[1].value
        weights=tf.get_variable('weights',shape=[dim,128],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases=tf.get_variable('biases',shape[128],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        local3=tf.nn.relu(tf.matmul(weights,reshape)+biases,name=scope.name)


    #全连接层4
    with tf.variable_scope('local4') as scope:
        weights=tf.get_variable('weights',shape=[128,128],dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases=tf.get_variable('biases',shape=[128],dtype=tf.float32,initializer=tf.constant_initialzier(0.1))
        local4=tf.nn.relu(tf.matmul(weights,local3)+biases,name=scope.name)
    #128个神经元，激活函数relu()
    
        #softmax
    #将前面的FC层输出，做一个线性回归，计算出每一类的得分，在这里是2类，所以这个层输出的是两个得分。
    with tf.variable_scope('softmax_linear') as scope:
        weights=tf.get_variable('weights',shape=[128,n_classes],dtype=tf.float32,
                               initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases=tf.get_variable('biases',shape=[n_classes],dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        softmax_linear=tf.add(tf.matmul(weights,local4),biases,name='softmax_linear')
        
    return softmax_linear  #linear 线性的


# In[27]:


#loss计算以及优化
#传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
#返回参数：loss，损失值
def losses(logits,labels):
    with tf.variable_scope('loss') as scope:
        #返回值是向量，值越小预测越准确
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                    labels=labels,name='xentropy_per_example')
        loss=tf.reduce_mean(cross_entropy,name='loss') #求均值
        tf.summary.scalar(scope.name+'loss',loss) #对标量数据汇总和记录
    return loss

def trainning(loss,learning_rate):
    #tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
    #tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量
    with tf.name_scope('optimizer'):
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step=tf.Variable(0,name='global_step',trainable=False)
        #优化更新训练的模型参数，也可以为全局步骤(global step)计数
        train_op=optimizer.minimize(loss,global_step=global_step) 
    return train_op

def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        #tf.nn.in_top_k组要是用于计算预测的结果和实际结果的是否相等，返回一个bool类型的张量
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy


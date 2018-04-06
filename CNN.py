#stride=步長，padding=抽離信息（特徵）的方式，pooling=整合特徵，減少特徵和參數
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

def weight_variable(shape):
  #shape表示生成张量的维度，mean是均值，stddev是标准差--随机数
  initial=tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial=tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
# strides[1,x_movement,y_movement,1]
# padding 提取方式
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
  
def max_pool_2x2(x):
#ksize 池化窗口的大小，四位向量
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


xs=tf.placeholder(tf.float32,[None,784])#28＊28
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,28,28,1])#-1:導入數據有多少個圖片，28*28*1:一個圖片像素點


#conv1 layer:卷积层
W_conv1=weight_variable([5,5,1,32])#patch 5*5,in size 1:图片厚度,out size32:卷积核数量
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28x28x32
h_pool1=max_pool_2x2(h_conv1)#output size 14x14x32

#conv2 layer
W_conv2=weight_variable([5,5,32,64])#patch 5*5,in size 32:图片厚度,out size64卷积核数量
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14x14x64
h_pool2=max_pool_2x2(h_conv2)#output size 7x7x64

#上面几步将输入转换为之前使用的方式
##func1 layer
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])
#[n_samples,7,7,64]->>[n_samples,7*7*64]
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)#降维度
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)#dropout是CNN中防止过拟合提高效果


##func2 layer
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

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
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
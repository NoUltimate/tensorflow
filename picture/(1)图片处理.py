
# coding: utf-8

# In[2]:


from PIL import Image
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[13]:


#step1 生成图片路径和标签的List
#路径

def get_files(file_dir):
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    zero=[]
    label_zero=[]
    first=[]
    label_first=[]
    second=[]
    label_second=[]
    third=[]
    label_third=[]
    fourth=[]
    label_fourth=[]
    for file in os.listdir(file_dir+'/zero'):
        zero.append(file_dir+'/zero'+'/'+file)
        label_zero.append(0)
    for file in os.listdir(file_dir+'/first'):
        first.append(file_dir+'/first/'+file)
        label_first.append(1)
    for file in os.listdir(file_dir+'/second'):
        second.append(file_dir+'/second/'+file)
        label_second.append(2)
    for file in os.listdir(file_dir+'/third'):
        third.append(file_dir+'/third/'+file)
        label_third.append(3)
    for file in os.listdir(file_dir+'/fourth'):
        fourth.append(file_dir+'/fourth/'+file)
        label_fourth.append(4)
    #step 2 对生成的图片路径和标签List做打乱处理
    image_list=np.hstack((zero,first,second,third,fourth))
    label_list=np.hstack((label_zero,label_first,label_second,label_third,label_fourth))

    #利用shuffle打乱顺序
    temp=np.array([image_list,label_list])
    temp=temp.transpose() #转置矩阵
    np.random.shuffle(temp)

    image_list=list(temp[:,0]) #X[:,0]就是取所有行的第0个数据
    label_list=list(temp[:,1])
    label_list=[int(i) for i in label_list]

    return image_list,label_list


# In[10]:


#生成batch
#image_W, image_H, ：设置好固定的图像高度和宽度
#设置batch_size：每个batch要放多少张图片
#capacity：一个队列最大多少

def get_batch(image,label,image_w,image_h,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)
    
    #make an input queue
    input_queue=tf.train.slice_input_producer([image,label])
    
    label=input_queue[1]
    #tf.read_file(string)
    image_contents=tf.read_file(input_queue[0]) #read img from a queue

    #step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image=tf.image.decode_jpeg(image_contents,channels=3)
    #step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image=tf.image.resize_image_with_crop_or_pad(image,image_w,image_h)

    image = tf.image.per_image_standardization(image)
    
    #step4：生成batch
    image_batch,label_batch=tf.train.batch([image,label],batch_size=batch_size,num_threads=32,capacity=capacity)
    
    #重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    #image_batch = tf.cast(image_batch, tf.float32)

    return image_batch,label_batch


# In[14]:


#测试
BATCH_SIZE = 6
CAPACITY = 256
IMG_W = 640
IMG_H = 480

train_dir='C:/light-faith/Github/data'

image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator() #创建一个协调器，管理线程 
    threads = tf.train.start_queue_runners(coord=coord) #启动QueueRunner, 此时文件名队列已经进队。 

    try:
        while not coord.should_stop() and i<2:

            img, label = sess.run([image_batch, label_batch])

            # just test one batch
            for j in np.arange(BATCH_SIZE):
                print('label: %d' %label[j])
                plt.imshow(img[j,:,:,:])
                plt.show()
            i+=1

    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop() #线程停止
    coord.join(threads)


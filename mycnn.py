
# coding: utf-8

# In[4]:


from PIL import Image
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[9]:


#step1 生成图片路径和标签的List
#路径
train_dir='C:/light-faith/Github/data'

def get_files(file_dir):
    # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    for file in os.listdir(file_dir+'/zero'):
        first.append(file_dir+'/zero/'+file)
        label_zero.append(0)
    for file in os.listdir(file_dir+'/first'):
        first.append(file_dir+'/first/'+file)
        label_first.append(1)
    for file in os.listdir(file_dir+'/second'):
        first.append(file_dir+'/second/'+file)
        label_second.append(2)
    for file in os.listdir(file_dir+'/third'):
        first.append(file_dir+'/third/'+file)
        label_third.append(3)
    for file in os.listdir(file_dir+'/fourth'):
        first.append(file_dir+'/fourth/'+file)
        label_fourth.append(4)


# In[11]:


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


# In[ ]:


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
    image_contents=tf.read_file(input_queue[0]) #read img from a queue


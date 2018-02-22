#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:56:08 2018

@author: TH
"""

import dataset
import tensorflow as tf
import time
import numpy as np
import math
import random
from datetime import timedelta
from numpy.random import seed

#把一个某个随机数编为10,便于后面再次使用
#每次使用时指定编码，随机数都是相同的
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)
batch_size=32
classes=["dogs","cats"]       #分类，从分类类生成编码
num_classes=len(classes)      #获取分类的数量


validation_size=0.2           #指定验证级的比例
img_size=64                   #指定图形的大小

#彩色图片是有三种颜色，颜色
#的深度是0-255,不同深度的
#不同颜色组合表示出艳丽的色
#彩。所以图片数组有三层，第
#一层表示三种基色，第二层表
#示高，第三层表示宽，里面的
#的数字表示基色的深度。
num_channels=3                #指定颜色的通道数


train_path="train_data"       #指定图片的目录


#使所有图片变成数组并读入内存中
data=dataset.read_train_sets(train_path,img_size,classes,validation_size=validation_size)
print("完成数据的读取")
print("训练集的文件数量为:{}".format(len(data.train.labels)))  #显示训练集有多少
print("验证集的文件数量为:{}".format(len(data.valid.labels)))  #显示测试集有多少


#定义会话
session=tf.Session()

#定义输入的数据集
#tensorflow要求是四维的因为
#第一个表示图片的索引
#第二个和第三个表示图片的长宽
#第三个表示通道数（就是是否三基色表达的）
#里面的元素就是对应的颜色深度
#总结：表达彩色图片要三维，还有一个是图片的
#索引所以要四维的素数
#第一个是批次数，最后一个是通道数。
x=tf.placeholder(tf.float32,shape=[None,img_size,img_size,num_channels],name="x")

#定义最后预测的数据
y_true=tf.placeholder(tf.float32,shape=[None,num_classes],name="y_true")

#定义预测的类型
#通过获取列位上数据最大的下标来定位类型
#如[0.1,0.9]表示是狗的概率是0.1,是猫
#的概率是0.9那么找到最大的概率的下标
#来定位["狗","猫"]的位置
y_true_cls=tf.argmax(y_true,dimension=1)


filter_size_conv1=3         #第一次卷积核是3x3的
num_filters_conv1=32        #第一次生成32个特征

filter_size_conv2=3         #第一次卷积核是3x3的
num_filters_conv2=32        #第一次生成32个特征

filter_size_conv3=3         #第一次卷积核是3x3的
num_filters_conv3=64        #第一次生成32个特征

fc_layer_size=1024          #全链接的大小



#根据传入的shape生成权重
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))

#根据传入的size生成偏置数
def create_biases(size):
    return tf.Variable(tf.constant(0.05,shape=[size]))

#针对一批次图片问题定义卷积函数
def create_convolutional_layer(input,               #输入的数据
                               num_input_channels,  #输入的通达
                               conv_filter_size,    #卷积核大小
                               num_filters):        #卷积特征数量，由于每次输入一定批次的数据，所以要一定数量的卷积，得到相同数量的特征
                               
    #由于数组是从里向外定义的，
    #读取却是从外向里读取
    #所以定义步奏为：
    #1、卷积核数长宽
    #2、通道数
    #3、特征数
    #卷积过程是：
    #取第一个特征的第一个通道的矩阵区域
    weights=create_weights(shape=[conv_filter_size,conv_filter_size,num_input_channels,num_filters])
    
    #定义一个相同特征的数量的偏置量
    biases=create_biases(num_filters)
    

    #进行卷积
    #不是书本上的矩阵乘
    #而是通过位置索引来找到对应数据的乘
    layer=tf.nn.conv2d(input=input,
                       filter=weights, 
                       #猜想前后的1表示，通道，和特征  
                       #因为偏置量是一张图片的     
                       strides=[1,1,1,1],     #中间的两个1表示长和宽的步长,前后的1必须是1
                       padding="SAME")        #padding表示在边缘没有数据时使用什么填充
    layer+=biases


    layer=tf.nn.relu(layer)                   #卷积后进行一次激励


    #进行池化，池化会缩小图片的大小
    layer=tf.nn.max_pool(value=layer,         #输入的数据
                         ksize=[1,2,2,1],     #定义池化核大小
                         strides=[1,2,2,1],   #输入的步长
                         padding="SAME")      
    return layer
    


#创建全链接层前的数据拉伸。是一个批次一个批次输入的
def create_flaten_layer(layer):

    #得到一个list。里面是四维数据的每一维的数据量
    #如运行后得到数据[32,8,8,3]
    layer_shape=layer.get_shape()   

    #由于第一个数据表示的是图片所有。我要的是图片的宽度，
    #高度，和通道数因此取1-3位置的数据,[1:4]表示1-3，4不取
    #后面的num_elements()是得到数据数量如取的是[8,8,3]
    #数据量就是8*8*3
    num_features=layer_shape[1:4].num_elements()  
    layer=tf.reshape(layer,[-1,num_features])
    
    return layer



def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights=create_weights(shape=[num_inputs,num_outputs])
    biases=create_biases(num_outputs)
    
    layer=tf.matmul(input,weights)+biases
    layer=tf.nn.dropout(layer,keep_prob=0.7)
    if use_relu:
        layer=tf.nn.relu(layer)
    return layer


layer_conv1=create_convolutional_layer(input=x,
                                     num_input_channels=num_channels,
                                     conv_filter_size=filter_size_conv1,
                                     num_filters=num_filters_conv1
                                     )

layer_conv2=create_convolutional_layer(input=layer_conv1,
                                     num_input_channels=num_filters_conv1,
                                     conv_filter_size=filter_size_conv2,
                                     num_filters=num_filters_conv2
                                     )

layer_conv3=create_convolutional_layer(input=layer_conv2,
                                     num_input_channels=num_filters_conv2,
                                     conv_filter_size=filter_size_conv3,
                                     num_filters=num_filters_conv3
                                     )





layer_flat=create_flaten_layer(layer_conv3)

layer_fc1=create_fc_layer(input=layer_flat,
                          num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                          num_outputs=fc_layer_size,
                          use_relu=True
                          
                          )


layer_fc2=create_fc_layer(input=layer_fc1,
                          num_inputs=fc_layer_size,
                          num_outputs=num_classes,
                          use_relu=False
                          
                          )

y_pred=tf.nn.softmax(layer_fc2,name="y_pred")

y_pred_cls=tf.argmax(y_pred,dimension=1)

session.run(tf.global_variables_initializer())

cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                      labels=y_true)

cost=tf.reduce_mean(cross_entropy)
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction=tf.equal(y_pred_cls,y_true_cls)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

session.run(tf.global_variables_initializer())
def show_progress(epoch,feed_dict_train,feed_dict_validate,val_loss,i):
    acc=session.run(accuracy,feed_dict=feed_dict_train)
    val_acc=session.run(accuracy,feed_dict=feed_dict_validate)
    msg="训练输出{}--迭代输出{}--训练的概率{}--验证通过概率{}---验证通过损失值{}"
    print(msg.format(epoch+1,i,acc,val_acc,val_loss))
    

total_iterations=0
sav=tf.train.Saver()


def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations+num_iteration):
        x_batch,y_true_batch,_,cls_batch=data.train.next_batch(batch_size)
        x_valid_batch,y_valid_batch,_,valid_cls_btch=data.valid.next_batch(batch_size)
        feed_dict_tr={x:x_batch,
                      y_true:y_true_batch}
        feed_dict_val={x:x_valid_batch,
                       y_true:y_valid_batch}
        
        
        session.run(optimizer,feed_dict=feed_dict_tr)
        
        if i%int(data.train.num_examples/batch_size)==0:
            val_loss=session.run(cost,feed_dict=feed_dict_val)
            epoch=int(i/int(data.train.num_examples/batch_size))
            show_progress(epoch,feed_dict_tr,feed_dict_val,val_loss,i)
            if i==998:
                sav.save(session,"./data/ko.ckpt",global_step=i)
    total_iterations+=num_iteration
    
train(num_iteration=1000)
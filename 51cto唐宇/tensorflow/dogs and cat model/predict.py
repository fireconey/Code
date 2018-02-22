#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:46:36 2018

@author: TH
"""

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

image_size=64
num_channels=3

images=[]

path="train_data/dog.jpg"

#读入一张图片
image=cv2.imread(path)

#image是数据源，imag_size是图片大小
#后面的0，0是方法的比例。如果指定大小了
#后面的数据失效。最后面的数据是使用什么
#方法填充确实的数据（或裁剪的数据）
image=cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)

#使用列别包装成4维数据
images.append(image)
images=np.array(images,dtype=np.uint8)    #包装成数组
images=images.astype("float32")           #转换成32位数据
images=np.multiply(images,1.0/225.0)      #颜色深度(0-255)数据归一化

x_batch=images.reshape(1,image_size,image_size,num_channels)
sess=tf.Session()


#导入训练的计算图
saver=tf.train.import_meta_graph("./data/ko.ckpt-998.meta")


 #导入W参数，后面的是点data的文件
 #后面的点不需要了
saver.restore(sess,"./data/ko.ckpt-998")  

#使用默认的图形结构
graph=tf.get_default_graph()

#在模型中获取预测标签入口，是预测最后的结果
y_pred=graph.get_tensor_by_name("y_pred:0")




#在模型中得到输入口
x=graph.get_tensor_by_name("x:0")


#start 其实下面的可以不要用于feed_dic=y_true:y_test_images
#在模型中获取真实标签入口，便于传入一个分类的编号
y_true=graph.get_tensor_by_name("y_true:0")
#生成一个全为0的1行2列的数据
y_test_images=np.zeros((1,2))
#end 


feed_dict_testing={x:x_batch}
result=sess.run(y_pred,feed_dict=feed_dict_testing)
res_label=["dog","cat"]
index=result.argmax()
#根据最大概率的下标来确定属于谁
print(res_label[index],result)

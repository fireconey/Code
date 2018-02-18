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

path="cat.1.jpg"
image=cv2.imread(path)

image=cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
images.append(image)
images=np.array(images,dtype=np.uint8)
images=images.astype("float32")
images=np.multiply(images,1.0/225.0)

x_batch=images.reshape(1,image_size,image_size,num_channels)
sess=tf.Session()


saver=tf.train.import_meta_graph("./data/ko.ckpt")
saver.restore(sess,"./data/ko.ckpt")


graph=tf.get_default_graph()

y_pred=graph.get_tensor_by_name("y_pred:0")

x=graph.get_tensor_by_name("y_true:0")
y_test_images=np.zeros((1,2))

feed_dict_testing={x:x_batch,y_true:y_test_images}
result=sess.run(y_pred,feed_dict=feed_dict_testing)

res_label=["dog","cat"]
print(res_label[result.argmax()])

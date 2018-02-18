#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 20:23:50 2018

@author: TH
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_points=1000
vectors_set=[]
for i in range(num_points):
    x1=np.random.normal(0,0.55)
    y1=x1*0.1+0.3+np.random.normal(0,0.03)
    vectors_set.append([x1,y1])

x_data=[v[0] for v in vectors_set]
y_data=[v[1] for v in vectors_set]


plt.show()
#在-1到1之间随机选取数，其中【1】表示外形
W=tf.Variable(tf.random_uniform([1],-1,1),name="w")
b=tf.Variable(tf.zeros([1]))
y=W*x_data+b
loss=tf.reduce_mean(tf.square(y-y_data))
#就是梯度安装0.5的改变
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
print("w=",sess.run(W),"b=",sess.run(b),"loss=",sess.run(loss))

for step in range(200):
    sess.run(train)
    print("w=",sess.run(W),"b=",sess.run(b),"loss",sess.run(loss))


#后面的c表示颜色，下面画的是点图
plt.scatter(x_data,y_data,c="r")
#下面的是线图
plt.plot(x_data,sess.run(W)*x_data+sess.run(b))
plt.show()


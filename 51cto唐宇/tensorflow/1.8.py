#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:39:13 2018

@author: TH

自己定义卷积函数
"""
import tensorflow as tf
#下载有input_data.py文件
import input_data
#一般由于数据集不能下载，自己下载后放在data文件夹中
mnist=input_data.read_data_sets("data/",one_hot=True)

#由于是图形计算所以要定义图形模板
#默认的图形可以在没有定义完操作前启动图形
tf.reset_default_graph()
#不使用Session是因为可以不使用session.run()
sess=tf.InteractiveSession()
#卷积必须是矩阵，后面的28，28表示输入必须是28*28的，1表示1通道
#None表示任意个，是一次性输入多少图片决定的。
x=tf.placeholder("float",shape=[None,28,28,1])
#预测的标签是10维的（数据分析中的维都表示的是列的数目）
y_=tf.placeholder("float",shape=[None,10])

#5，5表示卷积核大小,32表示使用了32种不同的卷积核，1表示通道
#会得到32中特征
w_conv1=tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1))
b_conv1=tf.Variable(tf.constant(.1,shape=[32]))

#以上是定义的卷积核
#以下是卷积
#strides=[1,1,1,1]表示移步大小。二三个表示长宽，第一个表示通道，第四个表示特征的
#由于卷积核移动到边缘后可能数据不足，所以使用padding="Same"使用相同的数据填充
h_conv1=tf.nn.conv2d(input=x,filter=w_conv1,strides=[1,1,1,1],padding="SAME")+b_conv1
h_conv1=tf.nn.relu(h_conv1)

#下面是磁化层
#ksize=【1,2,2,1】表示,2,2表示池化范围长宽，第一个表示通道，四个表示特征
#ksize表示池化的大小是个思维的
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



#卷积把上面的写成函数
def conv2d(x,w):
    return tf.nn.conv2d(input=x,filter=w,strides=[1,1,1,1],padding="SAME")

#池化写成函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")


#第二层使用了第一层数据
#5x5表示卷积核大小，32表示颜色通道也就是上面重叠的数据样本，64表示要得到的数据（每一层64个）
w_conv2=tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)


#以下是全链接层
#7*7由于单层通过2次池化（池化大小是2*2的大小，每步是2），所以长宽
#是原来的四分之一28/4=7，64表示有64层（得到了64个特征）1024表示拉成1024的
#由于最后一次池化后拉长后变成1x(7*7*64)矩阵要变成1x1024
#则要乘（7*7*64）x1024的矩阵就变成了1x1024
#1x(7*7*64)乘7*7*64）x1024=1x1024，相当于1x(7*7*64)是x 1x1024是w
w_fc1=tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
b_fc1=tf.Variable(tf.constant(.1,shape=[1024]))


#以下是x*w+b的函数
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#避免过拟合从中随机取部分数据
#是在运行时获取的图片批次中获取打个折扣
keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

#再次全链接
w_fc2=tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
b_fc2=tf.Variable(tf.constant(.1,shape=[10]))
y=tf.matmul(h_fc1_drop,w_fc2)+b_fc2


#交叉商
cross=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#自动调整括号中的参数不要指定0.5
train=tf.train.AdamOptimizer().minimize(cross)
#tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
#tf.argmax()表示矩阵(一定为矩阵)中那个最大值的下标位置
#后面一个参数表示方向，1表示横向，0表示纵向
#最后得到的是10维数据，同时有50(输入的图片数量)行
predict=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
acc=tf.reduce_mean(tf.cast(predict,"float"))
sess.run(tf.global_variables_initializer())
bat=50
for i in range(1000):
    ba=mnist.train.next_batch(bat)
    traininput=ba[0].reshape([bat,28,28,1])
    trainlabel=ba[1]
    if i%100==0:
        #eval是取值
        trainacc=acc.eval(session=sess,feed_dict={x:traininput,y_:trainlabel,keep_prob:0.5})
        print(i,"*",trainacc)
    ty=predict.eval(session=sess,feed_dict={x:traininput,y_:trainlabel,keep_prob:1})
    
    train.run(session=sess,feed_dict={x:traininput,y_:trainlabel,keep_prob:0.5})
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 15:58:28 2018

@author: TH
"""
import numpy as np
import cv2
import glob
import os
from sklearn.utils import shuffle



#由于数据的类型不同或者数据分散，但是这几个数据又是
#强相关的数据所以使用一个类进行了一个包装。好比列不同的
#表格
#分别对训练集合测试集进行包装
#同时有nex_batch()函数可以获取相应的批次的数据
class DataSet(object):
    def __init__(self,images,labels,img_names,cls):
        self._num_examples=images.shape[0]
        
        self._images=images
        self._labels=labels
        self._img_names=img_names
        self._cls=cls
        self._epochs_done=0
        self._index_in_epoch=0

    #使用@property可以直接使用".对应属性"来获取值    
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    
    @property
    def epochs_done(self):
        return self._epochs_done
    @property
    def num_examples(self):
        return self._num_examples
    
    def next_batch(self,batch_size):
        start=self._index_in_epoch
        self._index_in_epoch+=batch_size
        
        #向后依次去一批数据，如果数据不够一定数量就从头开始
        #***注意取的第一批的数据量一定要少于总数否则报错
        if self._index_in_epoch>self._num_examples:
            self._epochs_done+=1  #记录了循环了几次
            start=0
            self._index_in_epoch=batch_size
            
        end=self._index_in_epoch
        
        return self._images[start:end],self._labels[start:end],self._img_names[start:end],self._cls[start:end]
            
    

def read_train_sets(train_path,image_size,classes,validation_size):
    class DataSets(object):
        pass
    data_sets=DataSets()


    #调用load_train函数使文件夹下所以的图片读到内存中去
    images,labels,img_names,cls=load_train(train_path,image_size,classes)

    #进行洗牌的操作
    images,labels,img_names,cls=shuffle(images,labels,img_names,cls)
    

    #这里限制了验证级的大小指定必须是浮点数，通过比例来提取验证集的数量
    if isinstance(validation_size,float):
        validation_size=int(validation_size*images.shape[0])

    #从起始位置开始截取一定数据作为验证级
    validation_images=images[:validation_size]
    validation_labels=labels[:validation_size]
    validation_img_names=img_names[:validation_size]
    validation_cls=cls[:validation_size]
    

    #抛出验证数后获取的是训练级的数量
    train_images=images[validation_size:]
    train_labels=labels[validation_size:]
    train_img_names=img_names[validation_size:]
    train_cls=cls[validation_size:]
    

    #分别包装成训练集和测试集
    data_sets.train=DataSet(train_images,train_labels,train_img_names,train_cls)
    data_sets.valid=DataSet(validation_images,validation_labels,validation_img_names,validation_cls)
    
    return data_sets

    

        
        
    
    
    
    
def load_train(train_path,image_size,classes):
    images=[]
    labels=[]
    img_names=[]
    cls=[]
    print("读取图片")
    for fields in classes:
        index=classes.index(fields)
        print("现在读取的{}文件的下标为{}".format(fields,index))
        path=os.path.join(train_path,fields,"*g")
        files=glob.glob(path)
        for f1 in files:
            #导入的是一个数组
            image=cv2.imread(f1)
            #(image_size,image_size)是使用像素的形式指定大小
            #后面的0，0是比例的模式指定大小。只有有像素比例就默认失效
            #使像素失效，就设定为0，0


            #导入图片的数据
            image=cv2.resize(image,(image_size,image_size),0,0,cv2.INTER_LINEAR)
            image=image.astype(np.float32) #变成32为的
            image=np.multiply(image,1.0/255.0)#归一化操作
            images.append(image)

            #读入标签
            label=np.zeros(len(classes))  #生成[0,0]
            label[index]=1.0              #如果这个数据是dog的就在dog位置的0变成1
            labels.append(label)

            #读入文件名称
            flbase=os.path.basename(f1)   #获取文件的名称
            img_names.append(flbase)

            #读入文件的类别
        cls.append(fields)                #获取类别
    
    #列表变为数组不改变维度，由于元数据是
    #（图片）
    #三维的使用列表封装的，所以现在是四维
    #的数组,可以在第四维上通过索引来获取
    #对应的图片数组
    images=np.array(images)               #是列表变成数组
    labels=np.array(labels)               #标签变为数组
    img_names=np.array(img_names)         #名称变为数组
    cls=np.array(cls)                     #类别变为数组
    return images,labels,img_names,cls
    


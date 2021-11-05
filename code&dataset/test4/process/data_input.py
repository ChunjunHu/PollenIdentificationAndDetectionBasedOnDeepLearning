#!/usr/bin/env python
# encoding: utf-8
from keras.preprocessing.image import img_to_array, ImageDataGenerator#图片转为array
from keras.utils import to_categorical#相当于one-hot
from imutils import paths
import cv2
import numpy as np
import random
import os

# lastBatchsize32
def dataprocess(train_dir, validation_dir,height, width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size= batch_size,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size= 64, #RedDate 32 Polen 64
        class_mode='categorical')
    return train_generator, validation_generator



def load_data(path,norm_size,class_num):
    data = []#数据x
    label = []#标签y
    image_paths = sorted(list(paths.list_images(path)))#imutils模块中paths可以读取所有文件路径
    random.seed(0)#保证每次数据顺序一致
    random.shuffle(image_paths)#将所有的文件路径打乱
    for each_path in image_paths:
        image = cv2.imread(each_path)#读取文件
        image = cv2.resize(image,(norm_size,norm_size))#统一图片尺寸
        image = img_to_array(image)
        data.append(image)
        maker = int(each_path.split(os.path.sep)[-2])#切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        label.append(maker)
    data = np.array(data,dtype="float")/255.0#归一化
    label = np.array(label)
    label = to_categorical(label,num_classes=class_num)#one-hot
    return data,label

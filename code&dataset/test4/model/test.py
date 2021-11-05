#!/usr/bin/env python
# encoding: utf-8
import keras
import os
from keras import layers,models,optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import *   
from keras.models import Sequential, Model
import keras.backend as K
from keras.applications.xception import Xception,preprocess_input
from keras.applications.inception_v3 import InceptionV3
# from imutils import paths

# # train_dir = r"../data/dataset/train"
# validation_dir = r"../data/dataset/val"
# train_dir = r"E:\workspace\PicClass\test\traffic-master\data\dataset\train"

# global totalTrain = len(list(paths.list_images(basePath = train_dir)))
# global totalVal = len(list(paths.list_images(validation_dir)))
# # tuples = (1,2,3)


# # flag = paths.list_images(basePath = train_dir)
# # print(flag)

# # print(len(list(tuples)))

# print(totalTrain)
# print(totalVal)
# # print(list(paths.list_images(train_dir)))
# # list = [('a', 1),('b')]
# # print(list[0][1])



# input_shape = (300,400,3)
# conv_base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
# for layer in conv_base.layers:
#     layer.trainable = False
# conv_base.trainable = True
# # conv_base.trainable = False
# for layer in conv_base.layers:
#     print("{}: {}".format(layer, layer.trainable))



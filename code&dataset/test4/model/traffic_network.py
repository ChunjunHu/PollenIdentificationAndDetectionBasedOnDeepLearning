#!/usr/bin/env python
# encoding: utf-8
import keras
import os
from keras import layers,models,optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.layers import *   
from keras.models import Sequential, Model
import keras.backend as K
from keras import regularizers
from keras.applications.xception import Xception,preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3




class Lenet:#经典网络，不懂去查

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        model = Sequential()#顺序模型（keras中包括顺序模型和函数式API两种方式）
        model.add(Conv2D(16,(5,5),padding="same",activation="relu",input_shape=input_shape,name="conv1"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool1"))
        model.add(Conv2D(32,(5,5),padding="same",activation="relu",name="conv2",))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name="pool2"))
        model.add(Flatten())
        model.add(Dense(64,activation="relu",name="fc1"))
        model.add(Dense(classes,activation="softmax",name="fc2"))

        return model


class changXception:

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        conv_base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in conv_base.layers:
            layer.trainable = False
        x = conv_base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(classes, activation='softmax')(x)

        model = Model(inputs=conv_base.input, outputs=predictions)
        
        for layer in conv_base.layers:
            print("{}: {}".format(layer, layer.trainable))

        return model


class changXceptionUnfreezing:

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        conv_base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        for layer in conv_base.layers:
            layer.trainable = True
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block14_sepconv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False
        x = conv_base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(classes, activation='softmax')(x)

        model = Model(inputs=conv_base.input, outputs=predictions)
        
        for layer in conv_base.layers:
            print("{}: {}".format(layer, layer.trainable))
        
        return model

class changeVGG16:

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        # 模型微调
        #===================================================================================
        #-----------------------------------------------------------------------------------
        #载入VGG16网络, 进行特征提取
        # conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        # x = conv_base.output
        #-----------------------------------------------------------------------------------

        #First experiment
        # x = Flatten(name='flatten')(x)
        # x = Dense(256,activation='relu')(x)
        # x = Dropout(0.5)(x)
        #Second experiment 红枣数据集最终使用的上采样模型Before_After
        #x = Flatten(name='flatten')(x)
        #x = Dense(256, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)

        #Third experiment
        #x = Flatten(name='flatten')(x)
        #x = Dense(4096, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)
        #x = Dense(4096, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        #x = BatchNormalization()(x)
        #x = Dropout(0.5)(x)

        #predictions = Dense(classes, activation='softmax')(x)



        #top0
        # x = GlobalMaxPooling2D(name='GlobalMaxPooling2D')(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # predictions = Dense(classes, activation='softmax')(x)


        #top1
        # UpSample = MaxPooling2D(pool_size=(9, 9), strides=(1, 1),padding = 'same', name='MaxPooling2D')(x)
        # UpSample = Dropout(0.5)(UpSample)
        # UpSample = Conv2D(256,(1,1))(UpSample)
        # UpSample = BatchNormalization()(UpSample)
        # UpSample = Activation('relu')(UpSample)
        # UpSample = Dropout(0.5)(UpSample)
        # UpSample = Conv2D(256,(1,1))(UpSample)
        # UpSample = BatchNormalization()(UpSample)
        # UpSample = Activation('relu')(UpSample)
        # UpSample = Dropout(0.5)(UpSample)
        # UpSample = Flatten(name='flatten')(UpSample)
        # UpSample = Dense(classes)(UpSample)
        # UpSample = BatchNormalization()(UpSample)
        # predictions = Activation('softmax')(UpSample)
        #====================================================================================
        
        # 通过VGG16模型直接训练结果
        #==========================================================================================
        # version0
        # conv_base = VGG16(include_top=True, weights = None, input_shape=input_shape, classes = classes)
        # predictions = conv_base.output

        # version1
        conv_base = VGG16(include_top=False, weights = None, input_shape=input_shape)
        x = conv_base.output
        x = Flatten(name='flatten')(x)
        x = Dense(256,activation='relu')(x)
        x = Dense(256,activation='relu')(x)
        predictions = Dense(classes, activation='softmax')(x)   



        #==========================================================================================

        model = Model(inputs=conv_base.input, outputs=predictions)
        

        return model

class changeInceptionV3:

    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#确认输入维度
            input_shape = (height,width,channel)
        

        # x = GaussianNoise(0.3)(inputs)

        # 模型微调
        #=====================================================================================
        #--------------------------------------------------------------------------------------
        # 载入InceptionV3网络, 进行特征提取
        #conv_base = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
        #x = conv_base.output
        #--------------------------------------------------------------------------------------



        # # x = GlobalAveragePooling2D()(x)                1
        # # x = Dense(1024,activation='relu')(x)

        # # x = GlobalAveragePooling2D()(x)                2
        # # x = Dense(512)(x)
        # # x = BatchNormalization()(x)
        # # x = Activation('relu')(x)
        # # x = Dropout(0.5)(x)
        # # x = Dense(256)(x)
        # # x = BatchNormalization()(x)
        # # x = Activation('relu')(x)
        # # x = Dropout(0.5)(x)

        # # x = Flatten(name='flatten')(x)                 3
        # # x = Dense(256,activation='relu')(x)
        # # x = BatchNormalization()(x)
        # # x = Dropout(0.5)(x)


        # # x = GlobalAveragePooling2D()(x)                5
        # # x = Dropout(0.5)(x)
        # # x = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        # # x = BatchNormalization()(x)
        # # x = Dense(256, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        # # x = BatchNormalization()(x)
        # # x = Dropout(0.5)(x)

        # #x = GlobalAveragePooling2D()(x)                 4
        # #x = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        # #x = BatchNormalization()(x)
        # #x = Dropout(0.5)(x)
        # #x = Dense(512, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        # #x = BatchNormalization()(x)
        # #x = Dropout(0.5)(x)
        # #x = Dense(256, kernel_regularizer=regularizers.l2(0.01), activation='relu')(x)
        # #x = BatchNormalization()(x)
        # #x = Dropout(0.5)(x)

        # #top0
        # #UpSample = MaxPooling2D(pool_size=(8, 8), strides=(1, 1),padding = 'same')(x)
        # #UpSample = Dropout(0.5)(UpSample)
        # #UpSample = Conv2D(256,(1,1))(UpSample)
        # #UpSample = BatchNormalization()(UpSample)
        # #UpSample = Activation('relu')(UpSample)
        # #UpSample = Dropout(0.5)(UpSample)
        # #UpSample = Conv2D(256,(1,1))(UpSample)
        # #UpSample = BatchNormalization()(UpSample)
        # #UpSample = Activation('relu')(UpSample)
        # #UpSample = Dropout(0.5)(UpSample)
        # #UpSample = Flatten(name='flatten')(UpSample)
        # #UpSample = Dense(classes)(UpSample)
        # #UpSample = BatchNormalization()(UpSample)
        # #predictions = Activation('softmax')(UpSample)

        # #top1
        # x = Flatten(name='flatten')(x)
        # x = Dense(256,activation='relu')(x)
        # x = Dropout(0.5)(x)
        # predictions = Dense(classes, activation='softmax')(x)

        # #top2
        #x = GlobalMaxPooling2D(name='flatten')(x)
        #x = Dense(256,activation='relu')(x)
        #x = Dropout(0.5)(x)
        #predictions = Dense(classes, activation='softmax')(x)


        #==========================================================================================
        
        # 通过InceptionV3模型直接训练结果
        #==========================================================================================
        conv_base = InceptionV3(include_top=True, weights = None, input_shape=input_shape, classes = classes)
        predictions = conv_base.output



        #==========================================================================================
        model = Model(inputs=conv_base.input, outputs=predictions)

        return model



class selfNetwork:
    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#È·ÈÏÊäÈëÎ¬¶È
            input_shape = (height,width,channel)
        inputs = Input(shape= input_shape)
        x = Conv2D(32, (1, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (1, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)



        x4 = Conv2D(64, (1, 1), padding='same')(x)
        x4 = BatchNormalization()(x4)
        x4 = Activation('relu')(x4)

        x2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding = 'same')(x)
        x2 = Conv2D(32, (1, 1), padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x1 = Conv2D(96, (1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(128, (1, 3), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(128, (3, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Conv2D(128, (1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x3 = Conv2D(192, (1, 3), padding='same')(x)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (3, 1), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (1, 3), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (3, 1), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        # x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3)
        # x3 = Conv2D(256, (1, 3), padding='same')(x3)
        # x3 = BatchNormalization()(x3)
        # x3 = Activation('relu')(x3)
        # x3 = Conv2D(256, (1, 3), padding='same')(x3)
        # x3 = BatchNormalization()(x3)
        # x3 = Activation('relu')(x3)
        # x3 = Conv2D(256, (3, 1), padding='same')(x3)
        # x3 = BatchNormalization()(x3)
        # x3 = Activation('relu')(x3)
        # x3 = Conv2D(256, (1, 3), padding='same')(x3)
        # x3 = BatchNormalization()(x3)
        # x3 = Activation('relu')(x3)
        # x3 = Conv2D(256, (3, 1), padding='same')(x3)
        # x3 = BatchNormalization()(x3)
        # x3 = Activation('relu')(x3)
        # x3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x3)

        out = Concatenate(axis = -1)([x4, x1, x3, x2])

        #New
        #UpSample = MaxPooling2D(pool_size=(19, 19), strides=(1, 1),padding = 'same')(out)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Conv2D(256,(1,1))(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Activation('relu')(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Conv2D(256,(1,1))(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Activation('relu')(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Flatten(name='flatten')(UpSample)
        #UpSample = Dense(classes)(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #predictions = Activation('softmax')(UpSample)

        #New1
        UpSample = MaxPooling2D(pool_size=(19, 19), strides=(1, 1),padding = 'same')(out)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Conv2D(256,(1,1), kernel_regularizer=regularizers.l2(0.01))(UpSample)
        UpSample = BatchNormalization()(UpSample)
        UpSample = Activation('relu')(UpSample)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Conv2D(256,(1,1), kernel_regularizer=regularizers.l2(0.01))(UpSample)
        UpSample = BatchNormalization()(UpSample)
        UpSample = Activation('relu')(UpSample)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Flatten(name='flatten')(UpSample)
        UpSample = Dense(classes, kernel_regularizer=regularizers.l2(0.01))(UpSample)
        UpSample = BatchNormalization()(UpSample)
        predictions = Activation('softmax')(UpSample)

        #base
        #UpSample = Flatten(name='flatten')(out)
        #UpSample = Dense(128,activation='relu')(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #predictions = Dense(classes, activation='softmax')(UpSample)

        model = Model(inputs=inputs, outputs=predictions)

        
        return model

class selfNetwork_:
    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":#¨¨¡¤¨¨?¨º?¨¨????¨¨
            input_shape = (height,width,channel)
        inputs = Input(shape= input_shape)
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)
        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        out = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)
        #base
        UpSample = Flatten(name='flatten')(out)
        UpSample = Dense(128,activation='relu')(UpSample)
        UpSample = BatchNormalization()(UpSample)
        UpSample = Dropout(0.5)(UpSample)
        predictions = Dense(classes, activation='softmax')(UpSample)

        model = Model(inputs=inputs, outputs=predictions)

        
        return model

class selfNetworkMutilBlock:
    def neural(channel,height,width,classes):
        input_shape = (channel,height,width)
        if K.image_data_format() == "channels_last":
            input_shape = (height,width,channel)
        inputs = Input(shape= input_shape)
        x = Conv2D(32, (1, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (1, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(32, (3, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(x)



        x4 = Conv2D(64, (1, 1), padding='same')(x)
        x4 = BatchNormalization()(x4)
        x4 = Activation('relu')(x4)

        x2 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding = 'same')(x)
        x2 = Conv2D(32, (1, 1), padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)

        x1 = Conv2D(96, (1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(128, (1, 3), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Conv2D(128, (3, 1), padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x1 = Conv2D(128, (1, 1), padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x3 = Conv2D(192, (1, 3), padding='same')(x)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (3, 1), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (1, 3), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(192, (3, 1), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)

        out1 = Concatenate(axis = -1)([x4, x1, x3, x2])

        x1 = Conv2D(256, (3, 3), padding='same', strides = 2 )(out1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(64, (1, 1), padding='same')(out1)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(256, (3, 3), padding='same', strides = 2)(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
   

        x3 = Conv2D(128, (1, 1), padding='same')(out1)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(256, (3, 3), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(256, (3, 3), padding='same', strides = 2)(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)

        x4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same')(out1)

        out2 = Concatenate(axis = -1)([x1, x2, x3, x4])

        x1 = Conv2D(256, (1, 1), padding='same')(out2)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        x2 = Conv2D(768, (1, 1), padding='same')(out2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(640, (3, 3), padding='same')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Activation('relu')(x2)
   

        x3 = Conv2D(1044, (1, 1), padding='same')(out2)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(1024, (3, 3), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)
        x3 = Conv2D(1024, (1, 1), padding='same')(x3)
        x3 = BatchNormalization()(x3)
        x3 = Activation('relu')(x3)

        x4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding = 'same')(out2)
        x4 = Conv2D(128, (1, 1), padding='same')(x4)
        x4 = BatchNormalization()(x4)
        x4 = Activation('relu')(x4)

        out3 = Concatenate(axis = -1)([x1, x2, x3, x4])



        #top0
        UpSample = MaxPooling2D(pool_size=(10, 10), strides=(1, 1),padding = 'same')(out3)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Conv2D(256,(1,1))(UpSample)
        UpSample = BatchNormalization()(UpSample)
        UpSample = Activation('relu')(UpSample)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Conv2D(256,(1,1))(UpSample)
        UpSample = BatchNormalization()(UpSample)
        UpSample = Activation('relu')(UpSample)
        UpSample = Dropout(0.5)(UpSample)
        UpSample = Flatten(name='flatten')(UpSample)
        UpSample = Dense(classes)(UpSample)
        UpSample = BatchNormalization()(UpSample)
        predictions = Activation('softmax')(UpSample)

        #top1
        #UpSample = MaxPooling2D(pool_size=(10, 10), strides=(1, 1),padding = 'same')(out3)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Conv2D(256,(1,1), kernel_regularizer=regularizers.l1(0.01))(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Activation('relu')(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Conv2D(256,(1,1), kernel_regularizer=regularizers.l1(0.01))(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Activation('relu')(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #UpSample = Flatten(name='flatten')(UpSample)
        #UpSample = Dense(classes, kernel_regularizer=regularizers.l1(0.01))(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #predictions = Activation('softmax')(UpSample)


        #UpSample = Dense(256,activation='relu')(UpSample)
        #UpSample = BatchNormalization()(UpSample)
        #UpSample = Dropout(0.5)(UpSample)
        #predictions = Dense(classes, activation='softmax')(UpSample)

        model = Model(inputs=inputs, outputs=predictions)

        
        return model




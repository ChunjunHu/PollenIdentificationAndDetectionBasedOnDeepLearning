#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pylab as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import random
import sys
sys.path.append("../process")#添加其他文件夹
import data_input#导入其他模块
from traffic_network import Lenet
from traffic_network import changXception
from traffic_network import changXceptionUnfreezing
from traffic_network import changeVGG16
from traffic_network import changeInceptionV3
from traffic_network import selfNetwork
from traffic_network import selfNetwork_
from traffic_network import selfNetworkMutilBlock
from keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf
from tensorflow import keras as ke
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.initializers import glorot_normal
from tfdeterminism import patch
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD
#from keras.utils import plot_model
from imutils import paths
import os


#set Randomcode 
#---------------------------------------------------------------------------------------------------------
seed = 42
# using CPU to evaluate
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["PYTHONHASHSEED"] = str(seed)
#random.seed(seed)
#np.random.seed(seed)
#tf.random.set_random_seed(seed)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
#                              inter_op_parallelism_threads=1)
#tf.set_random_seed(seed)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

# using GPU to evaluate
patch()
os.environ['PYTHONHASHSEED']=str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['HOROVOD_FUSION_THRESHOLD']='0'
random.seed(seed)
np.random.seed(seed)
tf.random.set_random_seed(seed)
tf.set_random_seed(seed)

#--------------------------------------------------------------------------------------------------------
def train(model, train_generator, validation_generator, epochs, JudgeSaveRecord, batch_size):

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])#配置
    global totalTrain
    global totalVal
    #model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y))
    _history = model.fit_generator(
        train_generator,
        steps_per_epoch=totalTrain // batch_size,  #368
        epochs = epochs,
        validation_data=validation_generator,
        validation_steps=totalVal // 32)  #60

    history_dict = _history.history
    print(history_dict.keys())
    #dict_keys(['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'])
    #拟合，具体fit_generator请查阅其他文档,steps_per_epoch是每次迭代，需要迭代多少个batch_size，validation_data为test数据，直接做验证，不参与训练
    if JudgeSaveRecord:
        model.save("../predict/traffic_model.h5")
        plt.style.use("ggplot")#matplotlib的美化样式
        plt.figure()
        N = epochs
        plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")#model的history有四个属性，loss,val_loss,acc,val_acc
        plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
        plt.plot(np.arange(0,N),_history.history["acc"],label="train_acc")
        plt.plot(np.arange(0,N),_history.history["val_acc"],label="val_acc")
        plt.title("loss and accuracy")
        plt.xlabel("epoch")
        plt.ylabel("loss/acc")
        plt.legend(loc="best")
        plt.savefig("../result/result.png")
        plt.show()

# 定义特征提取
def setupTransferLearning(model, choOpt, freezedLayer):
    for layer in model.layers:
        layer.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == freezedLayer:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))
    model.compile(loss="categorical_crossentropy",
            optimizer=choOpt,metrics=["accuracy"])#配置

# 定义模型微调
def setupFineTune(model, choOpt, fineLayer):
    for layer in model.layers:
        layer.trainable = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == fineLayer:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
    for layer in model.layers:
        print("{}: {}".format(layer, layer.trainable))
    model.compile(loss="categorical_crossentropy",
            optimizer=choOpt,metrics=["accuracy"])#配置

# 绘制训练结果的图表
def plotResult(epochs, _history):
    plt.style.use("ggplot")#matplotlib的美化样式
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")#model的history有四个属性，loss,val_loss,acc,val_acc
    plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),_history.history["acc"],label="train_acc")
    plt.plot(np.arange(0,N),_history.history["val_acc"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("../result/result.png")
    plt.show()


if __name__ =="__main__":

    # num_cores = 4

    # if GPU:
    #     num_GPU = 1
    #     num_CPU = 1
    # if CPU:
    #     num_CPU = 1
    #     num_GPU = 0

    # config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
    #         inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
    #         device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    # session = tf.Session(config=config)
    # K.set_session(session)

    # with tf.compat.v1.Session() as ses:
    #     with tf.device("/cpu:0"):
    #         matrix1=tf.constant([[3.,3.]])
    #         matrix2=tf.constant([[2.],[2.]])
    #         product=tf.matmul(matrix1,matrix2)
    

    # 配置项目参数
    channel = 3
    height = 300 #300
    width = 300 #400
    class_num = 23 # 4
    #norm_size = 32#参数
    batch_size = 16
    epochs = 240
    sgd = SGD(lr=0.1,momentum=0.9) # 模型微调时使用 lr=1e-4, 模型直接训练时 lr=0.1
    rmsprop_ = 'rmsprop'
    FINE_LAYER = 'block5_conv1' # conv2d_73
    FREEZED_LAYER = 'MaxPooling2D' #flatten max_pooling2d_5


    train_dir = "../data/polenGroundTruth/train" # ../data/dataset/train
    validation_dir = "../data/polenGroundTruth/val" # ../data/dataset/val
    save_tl_dir = "../predict/TLCheckpoint"
    save_ft_dir = "../predict/FTCheckpoint"
    save_Direct_dir = "../predict/DirectCheckpoint"
    tl_best_model_path = os.path.join(save_tl_dir, 'TL_model_117-0.30.hdf5')

    totalTrain = len(list(paths.list_images(train_dir)))
    totalVal = len(list(paths.list_images(validation_dir)))
    print(totalTrain)

    NetworkName = "selfNetwork"

    JudgeFreezing = True
    JudgeSaveRecord = False
    switchCard = False # default:True, if change it to False, the network will not unfreezing after training with freezed conv_base


    # 开始训练
    print(">>>>>>>>>>>>>>>>>Begin the training<<<<<<<<<<<<<<<<<<")
    train_generator, validation_generator = data_input.dataprocess(train_dir, validation_dir, height, width, batch_size)
    #train_generator.reset()


    # 可视化输出处理后的数据的前三个样本
    # for i in range (3):
    #     x = next(train_generator)
    #     print(x)


    if NetworkName == "Xception":
        if JudgeFreezing == True:
            model = changXception.neural(channel=channel, height=height,
                                width=width, classes=class_num)#网络
            # model = Lenet.neural(channel=channel, height=height,
            #                      width=width, classes=class_num)#网络
            #train_x, train_y = data_input.load_data("../data/train", norm_size, class_num)
            #test_x, test_y = data_input.load_data("../data/test", norm_size, class_num)#生成数据

            # aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
            #                    height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
            #                    horizontal_flip=True,fill_mode="nearest")#数据增强，生成迭代器
            if switchCard == False:
                JudgeSaveRecord = True
            train(model,train_generator, validation_generator, epochs, JudgeSaveRecord, batch_size)#训练
            if switchCard == True:
                JudgeFreezing = False
                JudgeSaveRecord = True
        
        if JudgeFreezing == False:
            model = changXceptionUnfreezing.neural(channel=channel, height=height,
                                width=width, classes=class_num)#网络
            train(model,train_generator, validation_generator, epochs, JudgeSaveRecord, batch_size)#训练
    elif NetworkName == "VGG16":
        model = changeVGG16.neural(channel=channel, height=height,
                            width=width, classes=class_num)#网络
        model.summary()
        if switchCard == False:
            setupTransferLearning(model, rmsprop_, FREEZED_LAYER)
            checkpointerTL = ModelCheckpoint(os.path.join(save_tl_dir, 'TL_model_{epoch:02d}-{val_acc:.2f}.hdf5'),
                                   verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)
            _history = model.fit_generator(
                train_generator,
                steps_per_epoch=totalTrain // batch_size,
                epochs = epochs, 
                validation_data=validation_generator,
                validation_steps=totalVal // 64, # RedDate 32 Polen 64
                callbacks=[checkpointerTL])  
            history_dict = _history.history
            print(history_dict.keys())
        elif switchCard == True:
            model = load_model(tl_best_model_path)
            setupFineTune(model, sgd, FINE_LAYER)
            #Using tensorboard version
            tbCallback = TensorBoard(log_dir="../result/tbResult", histogram_freq=0, write_grads=True)
            checkpointerFT = ModelCheckpoint(os.path.join(save_ft_dir, 'FT_model_{epoch:02d}-{val_acc:.2f}.hdf5'),
                                   verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)
            _history = model.fit_generator(
                train_generator,
                steps_per_epoch=totalTrain // batch_size,
                epochs = epochs,
                validation_data=validation_generator,
                validation_steps=totalVal // 64,
                callbacks=[tbCallback, checkpointerFT]) 

            history_dict = _history.history
            print(history_dict.keys())
            plotResult(epochs, _history)

    # elif NetworkName == "InceptionV3":
    #     model, conv_base = changeInceptionV3.neural(channel=channel, height=height,
    #                 width=width, classes=class_num)#网络
    #     model.summary()
    #     setupTransferLearning(model, conv_base, rmsprop_)
    #     _history = model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=totalTrain // batch_size,  #368
    #         epochs = 103, #base:epochs, for different experiments we need do differ options
    #         validation_data=validation_generator,
    #         validation_steps=totalVal // 64)  # RedDate 32 Polen 64
    #     history_dict = _history.history
    #     print(history_dict.keys())
    #     model.save("../predict/RedDate_IV3_tl.h5")
    #     setupFineTune(model, conv_base, sgd, FINE_LAYER)
    #     #Using tensorboard version
    #     tbCallback = TensorBoard(log_dir="../result/tbResult", histogram_freq=0, write_grads=True)
    #     _history = model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=totalTrain // batch_size,  #368
    #         epochs = epochs,
    #         validation_data=validation_generator,
    #         validation_steps=totalVal // 64,
    #         callbacks=[tbCallback]) 

    #     history_dict = _history.history
    #     print(history_dict.keys())
    #     model.save("../predict/RedDate_IV3_fg.h5")
    #     plotResult(epochs, _history)

    elif NetworkName == "InceptionV3":
        model = changeInceptionV3.neural(channel=channel, height=height,
                    width=width, classes=class_num)#网络
        model.summary()
        if switchCard == False:
            setupTransferLearning(model, rmsprop_, FREEZED_LAYER)
            checkpointerTL = ModelCheckpoint(os.path.join(save_tl_dir, 'TL_model_{epoch:02d}-{val_acc:.2f}.hdf5'),
                                   verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)
            _history = model.fit_generator(
                train_generator,
                steps_per_epoch=totalTrain // batch_size,  #368
                epochs = epochs, 
                validation_data=validation_generator,
                validation_steps=totalVal // 64, # RedDate 32 Polen 64
                callbacks=[checkpointerTL])  
            history_dict = _history.history
            print(history_dict.keys())
        elif switchCard == True:
            model = load_model(tl_best_model_path)
            setupFineTune(model, sgd, FINE_LAYER)
            #Using tensorboard version
            tbCallback = TensorBoard(log_dir="../result/tbResult", histogram_freq=0, write_grads=True)
            checkpointerFT = ModelCheckpoint(os.path.join(save_ft_dir, 'FT_model_{epoch:02d}-{val_acc:.2f}.hdf5'),
                                   verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)
            _history = model.fit_generator(
                train_generator,
                steps_per_epoch=totalTrain // batch_size,  #368
                epochs = epochs,
                validation_data=validation_generator,
                validation_steps=totalVal // 64,
                callbacks=[tbCallback, checkpointerFT]) 

            history_dict = _history.history
            print(history_dict.keys())
            plotResult(epochs, _history)

    elif NetworkName == "selfNetwork":
        model = changeVGG16.neural(channel=channel, height=height,
                    width=width, classes=class_num)
        model.summary()
        model.compile(loss="categorical_crossentropy",
                optimizer=sgd,metrics=["accuracy"])# 最终决定使用sgd进行直接训练 学习率为0.1 比使用rmsprop效果更好
        #Using tensorboard version
        tbCallback = TensorBoard(log_dir="../result/tbResult", histogram_freq=0, write_grads=True)
        checkpointerFT = ModelCheckpoint(os.path.join(save_Direct_dir, 'Direct_model_{epoch:02d}-{val_acc:.2f}.hdf5'),
                                verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, period=1)
        _history = model.fit_generator(
            train_generator,
            steps_per_epoch=totalTrain // batch_size,  #368
            epochs = epochs,
            validation_data=validation_generator,
            validation_steps=totalVal // 64,   #60
            callbacks=[tbCallback, checkpointerFT])  
        #No tensorboard version
        #_history = model.fit_generator(
        #    train_generator,
        #    steps_per_epoch=totalTrain // batch_size,  #368
        #    epochs = epochs,
        #    validation_data=validation_generator,
        #    validation_steps=totalVal // 64)  #60
        history_dict = _history.history
        print(history_dict.keys())
        plotResult(epochs, _history)









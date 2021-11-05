from keras.preprocessing.image import img_to_array, ImageDataGenerator#图片转为array
from keras.utils import to_categorical#相当于one-hot
from imutils import paths
import cv2
import numpy as np
import random
import os

validation_dir = "image/val"
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        batch_size= 64, #RedDate 32 Polen 64
        class_mode='categorical').class_indices
print(validation_generator)

# from numpy import *
# import operator
# b = [[1,2,3],[111]]
# r = [1,2,3]
# a = [[0] * 4]*len(b)
# c = [[1,2],[2],[3]]
# f = []
# d = [123, 123, 123]

# arr=[[]*4 for i in range(4)]

# # lista = [[0,0,0], [1,1,1]]
# # b = [0,0,0]
# # for i in lista:
# #     if operator.eq(lista[i],b) == True:
# #         lista.pop(i)

# ss, bb = c[0]

# print(ss, bb)

# print(c[0])

# if b[0]==r:
#     print("ssssssss")

# def aaa(a):
#     a = a + 1
#     print("www")
# def bbb():
#     a = 1
#     aaa(a)
#     print(a)
# bbb()



# f.append([1,2,3])
# f.append([1,2,3])
# print(f[0])

# print(max(0, 10, 10))

# # #data5=mat(random.randint(2,8,size=(2,5))
# # #print(data5)
# # data5 = mat(random.randint(2,3,size=(2,5)))
# # print(data5)
# # print(tuple(data5))
# # print(str(data5))
# # #print(str(d))
# # print(list(str(d)))
# # print(len(c))
# # print(a)
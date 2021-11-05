import keras
from keras.models import load_model
# from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, ImageDataGenerator#图片转为array
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2


class predictProcess:
    def hhh(cropImage):
        classLableDict = {'anadenanthera': 0, 'arecaceae': 1, 'arrabidaea': 2, 'cecropia': 3, 'chromolaena': 4, 'combretum': 5, 'croton': 6, 'dipteryx': 7, 'eucalipto': 8, 'faramea': 9, 'hyptis': 10, 'mabea': 11, 'matayba': 12, 'mimosa': 13, 'myrcia': 14, 'protium': 15, 'qualea': 16, 'schinus': 17, 'senegalia': 18, 'serjania': 19, 'syagrus': 20, 'tridax': 21, 'urochloa': 22}
        data = []
        filepath = r'FT_model_71-0.97.hdf5'
        model = load_model(filepath)
        model.summary()
        #{'anadenanthera': 0, 'arecaceae': 1, 'arrabidaea': 2, 'cecropia': 3, 'chromolaena': 4, 'combretum': 5, 'croton': 6, 'dipteryx': 7, 'eucalipto': 8, 'faramea': 9, 'hyptis': 10, 'mabea': 11, 'matayba': 12, 'mimosa': 13, 'myrcia': 14, 'protium': 15, 'qualea': 16, 'schinus': 17, 'senegalia': 18, 'serjania': 19, 'syagrus': 20, 'tridax': 21, 'urochloa': 22}
        #img_path = 'image/myrcia_15.jpg'
        Cropimage = cv2.resize(cropImage,(300,300))#统一图片尺寸
        Cropimage = img_to_array(Cropimage)
        data.append(Cropimage)
        data = np.array(data,dtype="float")/255.0#归一化
        pred = model.predict(data)
        preds = np.argmax(pred, axis=1)
        print('Predicted:', preds)
        classLable = [k for k,v in classLableDict.items() if v==preds[0]]
        print(classLable)

        return classLable


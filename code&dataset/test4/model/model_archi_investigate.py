import os

from keras import layers,models,optimizers
from keras.applications.xception import Xception,preprocess_input
from keras.applications import VGG19
from keras.layers import *    
from keras.models import Model

base_model = Xception(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.summary()

from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
import numpy.random as rng
import numpy as np
import os
import dill as pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.datasets import cifar10
import sys

def W_init(shape,name=None):
    """Initialize weights as in paper"""
    values = rng.normal(loc=0,scale=1e-2,size=shape)
    return K.variable(values,name=name)
#//TODO: figure out how to initialize layer biases in keras.
def b_init(shape,name=None):
    """Initialize bias as in paper"""
    values=rng.normal(loc=0.5,scale=1e-2,size=shape)
    return K.variable(values,name=name)

class siamese(object):
    def __init__(self,img_size,img_channels):
        self.imsize=img_size
        self.img_channels=img_channels
        input_shape = (self.imsize, self.imsize, self.img_channels)
        left_input = Input(input_shape)
        right_input = Input(input_shape)
        #build convnet to use in each siamese 'leg'
        self.model = Sequential()

        self.model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(10, activation='relu'))
        #encode each of the two inputs into a vector with the convnet
        encoded_l = self.model(left_input)
        encoded_r = self.model(right_input)
        #merge two encoded inputs with the l1 distance between them
        L1_distance = lambda x: K.abs(x[0]-x[1])
        both = merge([encoded_l,encoded_r], mode = L1_distance, output_shape=lambda x: x[0])
        prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(both)
        self.siamese_net = Model(input=[left_input,right_input],output=prediction)
        #optimizer = SGD(0.0004,momentum=0.6,nesterov=True,decay=0.0003)

        optimizer = Adam(0.00006)
        #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
        self.siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

        self.siamese_net.count_params()
        #siamese_net.summary()
        print("Built and compiled siamese net")
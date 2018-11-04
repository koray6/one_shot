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

##########
#package imports 
from models import siamese
from utils import Siamese_Loader

def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    print("Loaded cifar10 from keras datasets, shape",x_train.shape )
    print("Shape of loaded data ",x_train.shape)
    loader=Siamese_Loader(x_train,x_test,y_train,y_test)
    sm=siamese(32,3)
    #sys.exit(0)
    evaluate_every = 7000
    loss_every=10
    batch_size = 32
    N_way = 20
    n_val = 550
    #siamese_net.load_weights("PATH")
    best = 76.0
    for i in range(900000):
        (inputs,targets)=loader.get_batch(batch_size)
        loss=sm.siamese_net.train_on_batch(inputs,targets)
        if i % evaluate_every == 0:
            #val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
            '''
            if val_acc >= best:
                print("saving")
                siamese_net.save('PATH')
                best=val_acc
            '''
            print("Network loss ",loss)

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))
if __name__ == '__main__':
    main()
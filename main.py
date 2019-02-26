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

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

gpu = '0'

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

##########
#package imports 
from models import siamese,siameseVAE
from utils import Siamese_Loader

from PIL import Image
import glob

def read_images(filename_occ,filename_unocc):
	image_list_train = []
	y_train=[]
	for filename in glob.glob('cal_data/'+filename_occ+'/*.png'): #assuming gif
	    im=Image.open(filename)
	    im = im.resize((32, 32))
	    im=np.array(im)
	    im=np.reshape(im,(32,32,1))
	    image_list_train.append(im)
	    y_train.append(1)
	
	for filename in glob.glob('cal_data/'+filename_unocc+'/*.png'): #assuming gif
	    im=Image.open(filename)
	    im = im.resize((32, 32))
	    im=np.array(im)
	    im=np.reshape(im,(32,32,1))
	    image_list_train.append(im)
	    y_train.append(0)
	x_train= np.array(image_list_train)
	y_train= np.array(y_train)
	y_train= np.reshape(y_train,(len(y_train),1))
	return x_train,y_train,x_train,y_train


def main():
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train, x_test, y_test=read_images('occupied','unoccupied')
    print("y_train shape ",y_train.shape)
    #sys.exit(0)
    #print("Shape of read images ",x_r.shape)
    #print("Shape of read images labels ",y_r.shape)
    #print("Loaded cifar10 from keras datasets, shape",x_train.shape )
    #print("Shape of loaded data ",x_train.shape)
    #print("Shape of loaded cifar data labels",y_train.shape)
    
    loader=Siamese_Loader(x_train,x_test,y_train,y_test)
    #loader.create_support(20)
    sm=siamese(32,1)

    # Just testing performance, make sure model loads the weights

    for filename in glob.glob('test_data/*.png'): #assuming gif
	    im=Image.open(filename)
	    im = im.resize((32, 32))
	    im=np.array(im)
	    im=np.reshape(im,(32,32,1))
	    loader.infer(sm.siamese_net,im,1)#Last argument doesnt matter
    #sm1=siameseVAE(28,1,200,50,32) #Need to try this for MNIST
    sys.exit(0)

    #This is the training part
    evaluate_every = 7000
    loss_every=500
    batch_size = 20
    N_way = 50
    n_val = 550
    #siamese_net.load_weights("PATH")
    best = 76.0
    for i in range(900000):
        (inputs,targets)=loader.get_batch(batch_size)
        loss=sm.siamese_net.train_on_batch(inputs,targets)
        if i % evaluate_every == 0:
            loader.test_oneshot_ability(sm.siamese_net,N_way)
            #sys.exit(0)
            #val_acc = loader.test_oneshot(siamese_net,N_way,n_val,verbose=True)
            '''
            if val_acc >= best:
                print("saving")
                siamese_net.save('PATH')
                best=val_acc
            '''
            print("Network loss ",loss)
            sm.siamese_net.save('model_weight')

        if i % loss_every == 0:
            print("iteration {}, training loss: {:.2f},".format(i,loss))
if __name__ == '__main__':
    main()
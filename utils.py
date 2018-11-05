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


class Siamese_Loader:
    """For loading batches and testing tasks to a siamese net"""
    def __init__(self,Xtrain,Xval,Ytrain,Yval): #Adapted slightly to load directly from cifar10
        numclasses=len(np.unique(Ytrain))
        self.Xtrain=np.zeros((numclasses,len(Xtrain),32,32,3))
        print("Got number of distinct classes as ",numclasses)
        print("Got new shape of Xtrain as ",self.Xtrain.shape)
        self.valid_tuple=[]
        for i in range(len(Ytrain)): #Warning !! not an efficient implementation, dictionary is recommended 
            self.Xtrain[Ytrain[i],i,:,:,:]=Xtrain[i]
            self.valid_tuple.append([Ytrain[i],i])
        #print("Got new Xtrain as ",self.Xtrain)
        self.Xval = Xval
        #self.Xtrain = Xtrain
        self.n_classes=numclasses
        self.n_examples,self.w,self.h,_ = Xtrain.shape
        self.n_val,self.n_ex_val,_,_ = Xval.shape

    def get_batch(self,n):
        """Create batch of n pairs, half same class, half different class"""
        categories = rng.choice(self.n_classes,size=(n,),replace=True)
        valid_choice=rng.choice(len(self.valid_tuple),size=(n,),replace=True)
        pairs=[np.zeros((n*n, self.h, self.w,3)) for i in range(2)] #Will create 2 dummy 0 image arrays

        #print("pairs look like ",pairs[0][1].shape)
        targets=np.zeros((n*n,)) #Placeholder for the labels of images
        #print("Shape of targets ",targets.shape)
        for i in range(int(len(valid_choice))):
            for j in range(int(len(valid_choice))):
                t1=self.valid_tuple[valid_choice[i]][0][0]
                t2=self.valid_tuple[valid_choice[j]][0][0]
                pairs[0][n*i+j,:,:,:] = self.Xtrain[self.valid_tuple[valid_choice[i]][0][0],self.valid_tuple[valid_choice[i]][1]].reshape(self.w,self.h,3)
                pairs[1][n*i+j,:,:,:] = self.Xtrain[self.valid_tuple[valid_choice[j]][0][0],self.valid_tuple[valid_choice[j]][1]].reshape(self.w,self.h,3)
                if(t1==t2):
                    targets[n*i+j]=1
                else:
                    targets[n*i+j]=0
        return pairs, targets

    def make_oneshot_task(self,N):
        """Create pairs of test image, support set for testing N way one-shot learning. """
        categories = rng.choice(self.n_val,size=(N,),replace=False)
        indices = rng.randint(0,self.n_ex_val,size=(N,))
        true_category = categories[0]
        ex1, ex2 = rng.choice(self.n_examples,replace=False,size=(2,))
        test_image = np.asarray([self.Xval[true_category,ex1,:,:]]*N).reshape(N,self.w,self.h,1)
        support_set = self.Xval[categories,indices,:,:]
        support_set[0,:,:] = self.Xval[true_category,ex2]
        support_set = support_set.reshape(N,self.w,self.h,1)
        pairs = [test_image,support_set]
        targets = np.zeros((N,))
        targets[0] = 1
        return pairs, targets

    def test_oneshot(self,model,N,k,verbose=0):
        """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
        pass
        n_correct = 0
        if verbose:
            print("Evaluating model on {} unique {} way one-shot learning tasks ...".format(k,N))
        for i in range(k):
            inputs, targets = self.make_oneshot_task(N)
            probs = model.predict(inputs)
            if np.argmax(probs) == 0:
                n_correct+=1
        percent_correct = (100.0*n_correct / k)
        if verbose:
            print("Got an average of {}% {} way one-shot learning accuracy".format(percent_correct,N))
        return percent_correct
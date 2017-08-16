import numpy as np
import os
from skimage import io
from resize_imgs import bulkResize

def celeba_process(x):
    x = x.astype(np.float32) / 255.0
    return x

#returns n images for training and n/5 images for testing
#x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).

def celeba_data(imageDirectory, n):
    xtrain=io.imread(imageDirectory+os.listdir(imageDirectory)[0])
    if xtrain.shape[0]!=32:
        bulkResize(imageDirectory, 32)
        imageDirectory=imageDirectory+ "/resized" + str(32) + "/"
        xtrain=io.imread(imageDirectory+os.listdir(imageDirectory)[0])
    xtrain=np.expand_dims(xtrain, axis=0)
    print "xtrain shape e'"
    print xtrain.shape
    for file in os.listdir(imageDirectory)[1:n]:
        img=np.expand_dims(io.imread(imageDirectory+file), axis=0) #(1,32,32,3)
        print "img shape e' " + str(img.shape)
        xtrain=np.concatenate((xtrain,img), axis=0)

    xtest=io.imread(imageDirectory+os.listdir(imageDirectory)[n])
    xtest=np.expand_dims(xtest, axis=0)
    for file in os.listdir(imageDirectory)[n+1:n+n/5]:
        img=np.expand_dims(io.imread(imageDirectory+file), axis=0) #(1,32,32,3)
        xtest=np.concatenate((xtest,img), axis=0)

    return celeba_process(xtrain), celeba_process(xtest)

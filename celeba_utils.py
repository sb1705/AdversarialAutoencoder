import numpy as np
import os
from skimage import io
from resize_imgs import bulkResize

def celeba_process(x):
    x = x.astype(np.float32) / 255.0
    return x

#returns n images for training and n/5 images for testing
#x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, size, size).
def celeba_data(imageDirectory, n, size, attr=0):
    xtrain=io.imread(imageDirectory+sorted(os.listdir(imageDirectory)[0]))
    xtest=0
    if xtrain.shape[0]!=size:
        print("inizio il resize")
        if(os.path.isdir(imageDirectory+ "/resized" + str(size) + "/")):
            print("Resize gia stato fatto")
        else:
            print("Ridimensiono le immagini")
            bulkResize(imageDirectory, size)
        imageDirectory=imageDirectory+ "/resized" + str(size) + "/"
        xtrain=io.imread(imageDirectory+sorted(os.listdir(imageDirectory)[0]))
    xtrain=np.expand_dims(xtrain, axis=0)
    for file in sorted(os.listdir(imageDirectory))[1:n]:
        img=np.expand_dims(io.imread(imageDirectory+file), axis=0) #(1,size,size,3)
        xtrain=np.concatenate((xtrain,img), axis=0)#(n,size,size,3)

    if(attr==0):
        xtest=io.imread(imageDirectory+sorted(os.listdir(imageDirectory))[n])
        xtest=np.expand_dims(xtest, axis=0)
        for file in sorted(os.listdir(imageDirectory))[n+1:n+n/5]:
            img=np.expand_dims(io.imread(imageDirectory+file), axis=0) #(1,size,size,3)
            xtest=np.concatenate((xtest,img), axis=0)#(n/5,size,size,3)

    return celeba_process(xtrain), celeba_process(xtest)

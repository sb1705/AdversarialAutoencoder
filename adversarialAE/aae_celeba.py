import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import sys
#from keras.layers import Reshape, Flatten, Lambda
#from keras.layers import Input
#from keras.layers.convolutional import UpSampling2D, MaxPooling2D
#from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #for tensorflow to work properly
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import l1l2, Dense, fit, Convolution2D, BatchNormalization
#from keras.layers.core import SpatialDropout2D
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
#from keras.layers import LeakyReLU, Activation
from scipy import ndimage, misc

from utils.image_utils import dim_ordering_unfix, dim_ordering_shape
from utils.data_utils import retrieve_data
from .nets import model_encoder, model_generator, model_generator
import sklearn
from sklearn import datasets


def AAE(output_path, shape, latent_width, color_channels, batch,
        epoch, image_path, n_imgs, adversarial_optimizer):
    # z in R^256
    latent_dim  = latent_width
    units       = 512
    # x in R^{3x64x64}
    img_size    = shape
    n_col       = color_channels
    input_shape = dim_ordering_shape((n_col, img_size, img_size))
    print("input shape:")
    print input_shape
    # generator (z -> x)
    generator = model_generator(latent_dim, n_col, units=units, dim_32=(img_size==32))
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape, units=units)
    # autoencoder (x -> x') --> join encoder e generator
    autoencoder = Model(encoder.inputs, generator(encoder(encoder.inputs)))
    # discriminator (z -> y)
    discriminator = model_discriminator(latent_dim, units=units)

    # build AAE
    x = encoder.inputs[0]
    z = encoder(x)
    xpred = generator(z)
    zreal = normal_latent_sampling((latent_dim,))(x)
    yreal = discriminator(zreal)
    yfake = discriminator(z)
    aae = Model(x, fix_names([xpred, yfake, yreal], ["xpred", "yfake", "yreal"]))

    # print summary of models
    print("AUTOENCODER : ENCODER + GENERATOR")
    autoencoder.summary()

    print("ENCODER")
    encoder.summary()

    print("GENERATOR")
    generator.summary()

    print("DISCRIMINATOR")
    discriminator.summary()


    # build adversarial model
    generative_params = generator.trainable_weights + encoder.trainable_weights
    model = AdversarialModel(base_model=aae,
                             player_params=[generative_params, discriminator.trainable_weights],
                             player_names=["A", "D"])
    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer,
                              player_optimizers=[Adam(3e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
                              loss={"yfake": "binary_crossentropy", "yreal": "binary_crossentropy",
                                    "xpred": "mean_squared_error"},
                              compile_kwargs={"loss_weights": {"yfake": 1e-1, "yreal": 1e-1, "xpred": 1e2}})

    # load data
    if(image_path=='olivetti'):
        data=sklearn.datasets.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)
        data=data['images']
        data=np.expand_dims(data, axis=3)
        xtrain=data[:300]
        xtest=data[300:]
        n = xtrain.shape[0]
    else:
        xtrain, xtest = retrieve_data(image_path, n_imgs, img_size)
        n = xtrain.shape[0]
        if(n_imgs == n):
           print "Dataset loading : Success"
        else:
           print "Dataset loading : Failure"

    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        generated=dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, img_size, img_size, n_col))
	if(n_col==1):
	    generated=np.squeeze(generated, axis=4)
	return generated

    generator_cb = ImageGridCallback(os.path.join(output_path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10) #(10,64,64,3)
        xrep = np.repeat(xsamples, 9, axis=0)#(90,64,64,3)
        xgen = dim_ordering_unfix(autoencoder.predict(xrep)).reshape((10, 9, n_col, img_size, img_size))
        xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, n_col, img_size, img_size))#(10,1,3,64,64)
        samples = np.concatenate((xsamples, xgen), axis=1)
        samples = samples.transpose((0, 1, 3, 4, 2))
	if(n_col==1):
            samples=np.squeeze(samples, axis=4)
        return samples

    if(n_col==1):
	autoencoder_cb = ImageGridCallback(os.path.join(output_path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler)
    else:
        autoencoder_cb = ImageGridCallback(os.path.join(output_path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler,
                                       cmap=None)


    y = [xtrain, np.ones((n, 1)), np.zeros((n, 1)), xtrain, np.zeros((n, 1)), np.ones((n, 1))]
    ntest = xtest.shape[0]
    ytest = [xtest, np.ones((ntest, 1)), np.zeros((ntest, 1)), xtest, np.zeros((ntest, 1)), np.ones((ntest, 1))]

    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest),
                  callbacks=[generator_cb, autoencoder_cb],
                  nb_epoch=epoch, batch_size=batch)

    # save history
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(output_path, "history.csv"))

    # save model
    encoder.save(os.path.join(output_path, "encoder.h5"))
    generator.save(os.path.join(output_path, "generator.h5"))
    discriminator.save(os.path.join(output_path, "discriminator.h5"))

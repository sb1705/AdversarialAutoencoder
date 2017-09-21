import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
import sys
from keras.layers import Reshape, Flatten, Lambda
from keras.layers import Input
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #for tensorflow to work properly
import keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import l1l2, Dense, fit, Convolution2D, BatchNormalization
from keras.layers.core import SpatialDropout2D
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from keras.layers import LeakyReLU, Activation
from scipy import ndimage, misc

from utils.image_utils import dim_ordering_unfix, dim_ordering_shape
from utils.data_utils import retrieve_data
#from .nets import model_encoder, model_generator, model_generator

def model_generator(latent_dim, units=512, dropout=0.5, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dim_32=False):
    model = Sequential(name="decoder")
    h = 5
    model.add(Dense(units * 4 * 4, input_dim=latent_dim, W_regularizer=reg()))
    model.add(Reshape(dim_ordering_shape((units, 4, 4))))
#    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
#    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
#    model.add(SpatialDropout2D(dropout))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 4, h, h, border_mode='same', W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    if not (dim_32):
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(units / 8, h, h, border_mode='same', W_regularizer=reg()))
        model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_encoder(latent_dim, input_shape, units=512, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.5):
    k = 5
    x = Input(input_shape)
    h = Convolution2D(units / 4, k, k, border_mode='same', W_regularizer=reg())(x)
#    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
#    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
#    h = SpatialDropout2D(dropout)(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units, k, k, border_mode='same', W_regularizer=reg())(h)
#    h = SpatialDropout2D(dropout)(h)
    h = LeakyReLU(0.2)(h)
    h = Flatten()(h)
    mu = Dense(latent_dim, name="encoder_mu", W_regularizer=reg())(h)
    log_sigma_sq = Dense(latent_dim, name="encoder_log_sigma_sq", W_regularizer=reg())(h)
    z = Lambda(lambda (_mu, _lss): _mu + K.random_normal(K.shape(_mu)) * K.exp(_lss / 2),
               output_shape=lambda (_mu, _lss): _mu)([mu, log_sigma_sq])
    return Model(x, z, name="encoder")


def model_discriminator(latent_dim, output_dim=1, units=256, reg=lambda: l1l2(1e-7, 1e-7)):
    z = Input((latent_dim,))
    h = z
#    mode = 1
    h = Dense(units, name="discriminator_h1", W_regularizer=reg())(h)
#    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(units / 2, name="discriminator_h2", W_regularizer=reg())(h)
#    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(units / 2, name="discriminator_h3", W_regularizer=reg())(h)
#    h = BatchNormalization(mode=mode)(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
    return Model(z, y)

def AAE(output_path, shape, latent_width, color_channels, batch,
        epoch, image_path, n_imgs, adversarial_optimizer):
#def AAE(image_path, n_imgs, output_path, adversarial_optimizer):
    # z \in R^256
    latent_dim  = latent_width
    units       = 512
    # x \in R^{3x64x64}
    img_size    = shape
    n_col       = color_channels
    input_shape = dim_ordering_shape((n_col, img_size, img_size))

    # generator (z -> x)
    generator = model_generator(latent_dim, units=units, dim_32=(img_size==32))
    # encoder (x ->z)
    encoder = model_encoder(latent_dim, input_shape, units=units)
    # autoencoder (x -> x') --> unisce encoder e generator
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

    xtrain, xtest = retrieve_data(image_path, n_imgs, img_size)


    # callback for image grid of generated samples
    def generator_sampler():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        return dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, img_size, img_size, 3))

    generator_cb = ImageGridCallback(os.path.join(output_path, "generated-epoch-{:03d}.png"), generator_sampler)

    # callback for image grid of autoencoded samples
    def autoencoder_sampler():
        xsamples = n_choice(xtest, 10) #(10,64,64,3)
        xrep = np.repeat(xsamples, 9, axis=0)#(90,64,64,3)
        #LAVORARE QUI
        xgen = dim_ordering_unfix(autoencoder.predict(xrep)).reshape((10, 9, 3, img_size, img_size))
        xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, 3, img_size, img_size))#(10,1,3,64,64)
        samples = np.concatenate((xsamples, xgen), axis=1)
        samples = samples.transpose((0, 1, 3, 4, 2))
        return samples

    autoencoder_cb = ImageGridCallback(os.path.join(output_path, "autoencoded-epoch-{:03d}.png"), autoencoder_sampler,
                                       cmap=None)

    # train network
    # generator, discriminator; pred, yfake, yreal
    n = xtrain.shape[0]
    if(n_imgs == n):
        print "Dataset loading : Success"
    else:
        print "Dataset loading : Failure"
        # print "n_imgs e' "+str(n_imgs)
        # print "shape e' "+ str(xtrain.shape)
        return
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


# def main():
#     if(len(sys.argv)<3):
#         print("Specify dataset's path and how many images you want to use for training")
#         return
#     else:
#         aae(sys.argv[1], int(sys.argv[2]), "output/aae"+str(sys.argv[2])+"img-"+str(500)+"ep-"+str(64)+"batchsize")
#
#
# if __name__ == "__main__":
#     main()

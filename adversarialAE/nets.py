from keras_adversarial.legacy import l1l2, Dense, fit, Convolution2D, BatchNormalization
#from keras.layers.core import SpatialDropout2D
from keras_adversarial import AdversarialModel, fix_names, n_choice
from keras.layers import Reshape, Flatten, Lambda
from keras.layers import Input
from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.layers import LeakyReLU, Activation
from keras.models import Sequential, Model

def model_generator(latent_dim, n_cols=3, units=512, dropout=0.5, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dim_32=False):
    model = Sequential(name="decoder")
    h = 5
    model.add(Dense(units * 4 * 4, input_dim=latent_dim, W_regularizer=reg()))
    model.add(Reshape(dim_ordering_shape((units, 4, 4))))
    model.add(LeakyReLU(0.2))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 2, h, h, border_mode='same', W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(units / 4, h, h, border_mode='same', W_regularizer=reg()))
    model.add(LeakyReLU(0.2))
    if not (dim_32):
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Convolution2D(units / 8, h, h, border_mode='same', W_regularizer=reg()))
        model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(n_cols, h, h, border_mode='same', W_regularizer=reg()))
    model.add(Activation('sigmoid'))
    return model


def model_encoder(latent_dim, input_shape, units=512, reg=lambda: l1l2(l1=1e-7, l2=1e-7), dropout=0.5):
    k = 5
    x = Input(input_shape)
    h = Convolution2D(units / 4, k, k, border_mode='same', W_regularizer=reg())(x)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units / 2, k, k, border_mode='same', W_regularizer=reg())(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = LeakyReLU(0.2)(h)
    h = Convolution2D(units, k, k, border_mode='same', W_regularizer=reg())(h)
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
    h = Dense(units, name="discriminator_h1", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(units / 2, name="discriminator_h2", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(units / 2, name="discriminator_h3", W_regularizer=reg())(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(output_dim, name="discriminator_y", activation="sigmoid", W_regularizer=reg())(h)
    return Model(z, y)

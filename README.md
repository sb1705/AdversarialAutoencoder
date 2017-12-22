# AdversarialAutoencoder

This repo contains a command line tool for the training of an adversarial autoencoder.

Keras (https://keras.io/) and Keras-adversarial (https://github.com/bstriner/keras-adversarial/) have been used for implementing the ANN.
Click (http://click.pocoo.org/5/) has been used for the command line support.

# Setup

For installing advae and it's dependencies, simply run:
```
python setup.py install
```

# Usage

With advae you can:
- train an Adversarial Autoencoder (command avae train)

![alt text](https://raw.githubusercontent.com/sb1705/AdversarialAutoencoder/new_master/advae_train.png)

- generate images using a already trained net (command andvae generate)

![alt text](https://raw.githubusercontent.com/sb1705/AdversarialAutoencoder/new_master/generate_help.png)

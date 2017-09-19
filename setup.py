from setuptools import setup

setup(
    name='adversarial_ae',
    version='1.0',
    author='Sara Brolli',
    author_email='',
    license='MIT',
    description='Python module for training an adversarial autoencoder on images.',
    packages=['adversarialAE'],
    long_description='',
    url='https://github.com/sb1705/AdversarialAutoencoder',
    keywords=['unsupervised', 'images', 'deep learning', 'neural networks', 'keras', 'adversarial', 'autoencoder'],
    install_requires=[
        'keras',
        'numpy',
        'click>=5.0',
        'matplotlib',
	    'tensorflow',
	    'pandas',
	    'scipy',
    ],
    entry_points={
        'console_scripts': [
            'adversarial_ae = adversarialAE.cli_tool:train'
        ],
    },
)

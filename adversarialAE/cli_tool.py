from time import strftime, localtime
from keras.models import load_model
import os
import numpy as np
from aae_celeba import AAE
from keras_adversarial import AdversarialOptimizerSimultaneous
#from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.image_grid import write_image_grid
from utils.image_utils import dim_ordering_unfix
import click

@click.command()
@click.option('--output-path', default='', type=click.Path(resolve_path=False,
              file_okay=False, dir_okay=True), 
              help='Output directory.')
@click.option('--shape', default=64, type = int,
              help='shape = image width = image_height. The possible values are 32 and 64. For example with shape=32 the images used for training will be (32, 32, number_of_colors)')
@click.option('--latent-width', default=256, type=int,
              help="Width of the latent space.")
@click.option('--color-channels', default=3, type=int,
              help='Number of colors.')
@click.option('--batch', default=64, type=int,
              help="Number of images per training batch.")
@click.option('--epoch', default=200, type=int,
              help="Number of epochs to train.")
@click.argument('image-path', default='olivetti', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.argument('n_imgs', default=500)
def train(output_path, shape, latent_width, color_channels, batch,
          epoch, image_path, n_imgs):
    '''Train an adversarial autoencoder on images'''
    if not ((shape==32)or(shape==64)):
	print "Images' shape must be 32 or 64"
	return
    if not (image_path == 'olivetti'):
        if not os.path.exists(os.path.dirname(image_path)):
            click.echo("Image path doesn't exist.")
            return
        if not image_path[-1] == '/':
            image_path += '/'
    else:
        color_channels=1

    if output_path=='':
        output_path='./output/'+strftime("%Y-%m-%d_%H:%M/", localtime())

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file = open(output_path+'summary.txt','w')
    file.write('dataset: ' + image_path+'\n')
    file.write('number of images: '+str(n_imgs)+'\n')
    file.write('output path: '+str(output_path)+'\n')
    file.write('image shape: ('
                              +      str(shape)
                              +',' + str(shape)
                              +',' + str(color_channels)
                              +')\n')
    file.write('width of latent space: '+str(n_imgs)+'\n')
    file.write('number of epochs: ' + str(epoch)+'\n')
    file.write('batch size: ' + str(batch)+'\n')
    file.close()


    AAE(str(output_path), int(shape), int(latent_width), int(color_channels), int(batch),
        int(epoch), str(image_path), int(n_imgs), AdversarialOptimizerSimultaneous())
    #AAE(str(image_path), int(n_imgs), str(output_path), AdversarialOptimizerSimultaneous())


@click.command()
@click.option('--output-path', default='', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.argument('path', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
def generate(path, output_path):
    """Generate 100 images from previously trained model"""
    if not path[-1] == '/':
        path += '/'
    if(output_path==''):
        output_path=str(path)+'generated/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print output_path
    generator = load_model(os.path.join(path, "generator.h5"))
    latent_dim=generator.input_shape[1]
    #out=generator.layers[-1]
    img_size=generator.output_shape[2]
    n_col=generator.output_shape[3]
    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    imgs=dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, img_size, img_size, n_col))
    if n_col == 1:
        imgs=np.squeeze(imgs, axis=4)
        write_image_grid(os.path.join(output_path, "generated.png"), imgs)
    else:
        write_image_grid(os.path.join(output_path, "generated.png"), imgs, cmap=None)    
#ImageGridCallback(os.path.join(output_path, "generated.png"), imgs)
    return
    # def generator_sampler():
    #     zsamples = np.random.normal(size=(10 * 10, latent_dim))
    #     return dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, img_size, img_size, n_col))
    #
    # ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)


#@click.command()
#@click.pass_context
#def help(ctx):
#    print(ctx.parent.get_help())


@click.group(context_settings={'help_option_names':['-h','--help']}, help='A tool for training an adversarial autoencoder')
def initcli():
    pass

initcli.add_command(train)
initcli.add_command(generate)
#initcli.add_command(help)

#if __name__ == '__main__':
#    initcli()

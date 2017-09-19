from time import gmtime, strftime
import os
from aae_celeba import AAE
from keras_adversarial import AdversarialOptimizerSimultaneous
import click

@click.command()
@click.option('--output-path', default='', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.option('--shape', default=64, type = int,
              help='shape = image width = image_height. For example with shape=32 the images used for training will be (32, 32, number_of_colors)')
@click.option('--latent-width', default=256, type=int,
              help="Width of the latent space.")
@click.option('--color-channels', default=3, type=int,
              help='Number of colors.')
@click.option('--batch', default=64, type=int,
              help="Number of images per training batch.")
@click.option('--epoch', default=200, type=int,
              help="Number of epochs to train.")
@click.argument('image-path', type=click.Path(resolve_path=False,
                                              file_okay=False, dir_okay=True))
@click.argument('n_imgs', default=500)
def train(output_path, shape, latent_width, color_channels, batch,
          epoch, image_path, n_imgs):
    '''Train'''
    if not os.path.exists(os.path.dirname(image_path)):
        click.echo("Image path doesn't exist.")
        return
    if not image_path[-1] == '/':
        image_path += '/'
    if output_path=='':
        output_path='/output/'+strftime("%Y-%m-%d_%H:%M", gmtime())
    else:
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

    file = open('summary.txt','w')
    file.write('dataset: ' + image_path)
    file.write('number of images: '+str(n_imgs))
    file.close()

    AAE(str(output_path), int(shape), int(latent_width), int(color_channels), int(batch),
        int(epoch), str(image_path), int(n_imgs), AdversarialOptimizerSimultaneous())
    #AAE(str(image_path), int(n_imgs), str(output_path), AdversarialOptimizerSimultaneous())


#@click.command()
#def generate():
#    return
    # def generator_sampler():
    #     zsamples = np.random.normal(size=(10 * 10, latent_dim))
    #     return dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1)).reshape((10, 10, img_size, img_size, n_col))
    #
    # ImageGridCallback(os.path.join(path, "generated-epoch-{:03d}.png"), generator_sampler)


#@click.command()
#@click.pass_context
#def help(ctx):
#    print(ctx.parent.get_help())


#@click.group(context_settings={'help_option_names':['-h','--help']}, help='A tool for training an adversarial autoencoder')
#def initcli():
#    pass

#initcli.add_command(train)
#initcli.add_command(generate)
#initcli.add_command(help)

#if __name__ == '__main__':
#    initcli()

from . import aae_celeba
from . import nets
from . import utils

AAE = aae_celeba.AAE

#model_generator = nets.model_generator
#model_encoder = nets.model_encoder
#model_discriminator = nets.model_discriminator

retrieve_data = utils.data_utils.retrieve_data
data_process = utils.data_utils.data_process

dim_ordering_fix = utils.image_utils.dim_ordering_fix
dim_ordering_unfix = utils.image_utils.dim_ordering_unfix
dim_ordering_shape = utils.image_utils.dim_ordering_shape
dim_ordering_shape = utils.image_utils.dim_ordering_shape
dim_ordering_input = utils.image_utils.dim_ordering_input
dim_ordering_reshape = utils.image_utils.dim_ordering_reshape
channel_axis = utils.image_utils.channel_axis

resize = utils.resize_imgs.resize
bulkResize = utils.resize_imgs.bulkResize

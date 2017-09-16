import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #for tensorflow to work properly
import numpy as np
from keras.models import load_model
from keras_adversarial.image_grid import write_image_grid
from celeba_utils import celeba_data
from scipy import ndimage, misc
from image_utils import dim_ordering_unfix
from keras_adversarial import AdversarialOptimizerSimultaneous
# returns a list with all attribute names & a list of array with at i-th position
# the attribute vector corrisponding at the i-th image.
def list_attr(attr_path):
    all_attributes = []
    with open(attr_path, 'r') as f:
        f.readline()
        attribute_names = f.readline().strip().split(' ')#lista dei nomi
        for i, line in enumerate(f):
            fields = line.strip().replace('  ', ' ').split(' ')
            img_name = fields[0]
            if int(img_name[:6]) != i + 1:
                raise ValueError('Parse error.')
            attr_vec = np.array(map(int, fields[1:]))
            all_attributes.append(attr_vec)
    return(attribute_names, all_attributes)

def load_models(path):
    encoder   = load_model(os.path.join(path, "encoder.h5"))
    generator = load_model(os.path.join(path, "generator.h5"))
    return(encoder, generator)


def add_attributes(data_path, model_path, attr_path):
    latent_dim = 256
    n_imgs = 200
    img_size=64
    imgs, imgs2 = celeba_data(data_path, n_imgs, 64, attr=1)

    #load attributes
    attribute_names=[]
    all_attributes=[]
    attribute_names, all_attributes = list_attr(attr_path)

    #load models
    encoder, generator = load_models(model_path)

    selected_attributes = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        'Mustache', 'Pale_Skin', 'Rosy_Cheeks', 'Smiling', 'Straight_Hair',
        'Wavy_Hair', 'Wearing_Lipstick', 'Young',
    ]

    #encode imgs
    z = (encoder.predict(imgs[:n_imgs]))
    print "zshape: "+ str(z.shape)

    #indexes of selected attributes
    attr_indxs = [attribute_names.index(attr) for attr in selected_attributes]

    attr_vecs = []#attr_vecs[i]=z vector of i-th attr
    for attr_indx in attr_indxs:
        #on_mask[i]==True if the i-th img has the attr_indx-th attribute
        on_mask = [x==1 for x in zip(*all_attributes)[attr_indx]]
        off_mask = np.logical_not(on_mask)
        vec = (np.mean(z[on_mask[:n_imgs]], dtype=float) -
               np.mean(z[off_mask[:n_imgs]], dtype=float))
        attr_vecs.append(vec)
    print "attr_vecs lenght :"
    print len(attr_vecs)

    # print imgs with attributes
    xsamples = imgs[:10] #(10,64,64,3)
    original_z = z[:10]
    new_z = np.repeat(original_z, 9, axis=0)
    
    for i in range(0, 9):
        new_z[i] += attr_vecs[i] #primi 9 attributi
        
    # xrep = np.repeat(xsamples, 9, axis=0)#(90,64,64,3)
    # print "xrep.shape :"
    # print xrep.shape
    # for i in range(1, 10):
    #     xrep[i] += attr_vecs[i-1] #primi 9 attributi
    
    # original_x = generator.predict(original_z)
    new_x = generator.predict(new_z)
    print "new_x shape  :"
    print new_x.shape
    xgen = dim_ordering_unfix(new_x).transpose((0, 2, 3, 1)).reshape((10, 9, 3, img_size, img_size))
    xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, 3, img_size, img_size))#(10,1,64,64,3)
    samples = np.concatenate((xsamples, xgen), axis=1)
    samples = samples.transpose((0, 1, 3, 4, 2))
    print "samples.shape :"
    print samples.shape
    write_image_grid(os.path.join(model_path, "attr.png"), samples, cmap=None)
    

def main():
    if(len(sys.argv)<4):
        print("Specify dataset's path, model's path, attribute file path")
        return
    else:
        add_attributes(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()

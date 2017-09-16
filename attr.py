import os
import numpy as np
from image_utils import dim_ordering_unfix

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
    encoder.save(os.path.join(path, "encoder.h5"))
    generator.save(os.path.join(path, "generator.h5"))
    return(encoder, generator)


def add_attributes(data_path, model_path, attr_path):
    latent_dim = 256
    n_imgs = 200
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

    #indexes of selected attributes
    attr_indxs = [attribute_names.index(attr) for attr in selected_attributes]

    attr_vecs = []#attr_vecs[i]=z vector of i-th attr
    for attr_indx in attr_indxs:
        #on_mask[i]==True if the i-th img has the attr_indx-th attribute
        on_mask = [x==1 for x in zip(*all_attributes)[attr_indx]]
        off_mask = np.logical_not(on_mask)
            vec = (np.mean(z[on_mask], dtype=float) -
                   np.mean(z[off_mask], dtype=float))
        attr_vecs.append(vec)
        print "attr_vecs :"
        print attr_vecs

    # print imgs with attributes
    xsamples = img[:10] #(10,64,64,3)
    xrep = np.repeat(xsamples, 9, axis=0)#(90,64,64,3)
    for i in range(1, 10)
        xrep[i] += vec[i-1] #primi 9 attributi
    xenc = dim_ordering_unfix(encoder.predict(xrep).reshape((10, 9, 3, latent_dim))
    xgen = dim_ordering_unfix(generator.predict(xenc)).transpose((0, 2, 3, 1)).reshape((10, 9, img_size, img_size, 3))
    xsamples = dim_ordering_unfix(xsamples).reshape((10, 1, img_size, img_size, 3))#(10,1,64,64,3)
    samples = np.concatenate((xsamples, xgen), axis=1)
    return samples

    write_image_grid(model_path, samples, cmap=None)

def main():
    if(len(sys.argv)<4):
        print("Specify dataset's path, model's path, attribute file path")
        return
    else:
        attribute(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), "output/attribute", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()

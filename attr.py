import os
import numpy as np
#from PIL import Image

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

    #indexes of selected attributes
    attr_indxs = [attribute_names.index(attr) for attr in selected_attributes]

    attr_vecs = []
for attr_indx in attr_indxs: #per ogni indice selezionato
    on_mask = all_attributes[:][attr_idx] == 1.0 #non funziona!
    off_mask = np.logical_not(on_mask)
        vec = (np.mean(z[on_mask, :], axis=0, dtype=float) -
               np.mean(z[off_mask, :], axis=0, dtype=float))
    attr_vecs.append(vec)
    print "attr_vecs :"
    print attr_vecs

def main():
    if(len(sys.argv)<4):
        print("Specify dataset's path, model's path, attribute file path")
        return
    else:
        attribute(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), "output/attribute", AdversarialOptimizerSimultaneous())


if __name__ == "__main__":
    main()

import os
import numpy as np
#from PIL import Image

#all_attributes = list(dp.dataset.CelebA().attribute_names)
attr_path="/home/sara/Documents/Tesi/AdversarialAutoencoder/dataset/list_attr_celeba.txt"
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
#attributes = np.array(attributes)
#    attribute_names=np.array(attribute_names)

    selected_attributes = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
        'Mustache', 'Pale_Skin', 'Rosy_Cheeks', 'Smiling', 'Straight_Hair',
        'Wavy_Hair', 'Wearing_Lipstick', 'Young',
    ]
    attr_idxs = [attribute_names.index(attr) for attr in selected_attributes] #inidici degli attributi selezionati
    print "attr_idxs :"
    print attr_idxs
    attr_vecs = []
    for attr_idx in attr_idxs:
        on_mask = y[:, attr_idx] == 1.0
        off_mask = np.logical_not(on_mask)
        vec = (np.mean(z[on_mask, :], axis=0, dtype=float) -
               np.mean(z[off_mask, :], axis=0, dtype=float))
        attr_vecs.append(vec)
    print "attr_vecs :"
    print attr_vecs

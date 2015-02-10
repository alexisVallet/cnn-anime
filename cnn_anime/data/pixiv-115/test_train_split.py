""" Makes a random test/validation/train split for the pixiv-115 dataset.
"""
import cPickle as pickle
import os
import os.path
import sys
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Please give an input tags file, split name, number of test samples and number of validation samples.")
    in_tags_fname = sys.argv[1]
    name = sys.argv[2]
    nb_test = int(sys.argv[3])
    nb_valid = int(sys.argv[4])
    in_tags = pickle.load(open(in_tags_fname, 'rb'))
    # Compute the raw tags dataset.
    fnames = []
    tags = []

    for folder in in_tags:
        for img_id in in_tags[folder]:
            fnames.append(os.path.join(
                folder.decode('utf-8'),
                repr(img_id) + '.jpg'
            ).encode('utf-8'))
            tags.append(in_tags[folder][img_id])

    # Then take a random permutation of that, split it.
    nb_samples = len(fnames)
    permut = np.random.permutation(nb_samples)
    i = 0
    test_set = {}

    while i < nb_test:
        test_set[fnames[permut[i]]] = tags[permut[i]]
        i += 1
    valid_set = {}
    
    while i < nb_test + nb_valid:
        valid_set[fnames[permut[i]]] = tags[permut[i]]
        i += 1

    train_set = {}
    while i < nb_samples:
        train_set[fnames[permut[i]]] = tags[permut[i]]
        i += 1

    # Pickling all that crap.
    pickle.dump(
        test_set,
        open(name + '_test.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    pickle.dump(
        valid_set,
        open(name + '_valid.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )
    pickle.dump(
        train_set,
        open(name + '_train.pkl', 'wb'),
        protocol=pickle.HIGHEST_PROTOCOL
    )

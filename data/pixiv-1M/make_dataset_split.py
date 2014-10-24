""" Generates a random train/validation/test split out of the pixiv-1M dataset.
    Because the set of images is huge an unwieldy and I don't want to touch it,
    this only operates on the filename/labels set.
"""
import cPickle as pickle
import numpy as np
import sys
import os.path
from random import shuffle
import cv2

if __name__ == "__main__":
    # Take as arguments size of training, validation and test dataset.
    if len(sys.argv) < 6:
        raise ValueError("Please input a filename for labels, sizes for training, validation and test datasets, and a name for the dataset.")
    labels_dict_fname = sys.argv[1]
    train_size = int(sys.argv[2])
    valid_size = int(sys.argv[3])
    test_size = int(sys.argv[4])
    total_size = train_size + valid_size + test_size
    dataset_name = sys.argv[5]
    # Takes as input the filename/labels dict. Generates 3 pickled
    # python dictionaries from filename to set of labels.

    assert os.path.isfile(labels_dict_fname)

    labels_dict = None

    with open(labels_dict_fname, 'rb') as labels_dict_file:
        labels_dict = pickle.load(labels_dict_file)

    fname_labels = labels_dict.values()
    # Remove corrupt images from the bunch.
    shuffle(fname_labels)
    fname_labels = map(lambda fandl: (os.path.basename(fandl[0]), fandl[1]), fname_labels)
    non_corrupt = []
    i = 0
    
    while len(non_corrupt) < total_size:
        image = cv2.imread('images/' + fname_labels[i][0])
        if image != None:
            non_corrupt.append(fname_labels[i])
        else:
            print repr(i) + " th image is corrupt: " + repr(fname_labels[i][0])
        i += 1
        
    test_set = {}
    i = 0
    while i < test_size:
        test_set[non_corrupt[i][0]] = non_corrupt[i][1]
        i += 1
    valid_set = {}
    while i < test_size + valid_size:
        valid_set[non_corrupt[i][0]] = non_corrupt[i][1]
        i += 1
    train_set = {}
    while i < test_size + valid_size + train_size:
        train_set[non_corrupt[i][0]] = non_corrupt[i][1]
        i += 1
    test_fname = dataset_name + '_test.pkl'
    valid_fname = dataset_name + '_valid.pkl'
    train_fname = dataset_name + '_train.pkl'

    with open(test_fname, 'w') as test_file:
        pickle.dump(test_set, test_file)
    with open(valid_fname, 'w') as valid_file:
        pickle.dump(valid_set, valid_file)
    with open(train_fname, 'w') as train_file:
        pickle.dump(train_set, train_file)


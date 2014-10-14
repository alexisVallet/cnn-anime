""" Generates a random train/validation/test split out of the pixiv-1M dataset.
    Because the set of images is huge an unwieldy and I don't want to touch it,
    this only operates on the filename/labels set.
"""
import cPickle as pickle
import numpy as np
import sys
import os.path
from random import shuffle

if __name__ == "__main__":
    # Use the same numbers for validation and test as ilsvrc 2014 classification.
    test_size = 150000
    valid_size = 50000
    # Takes as input the filename/labels dict. Generates 3 pickled
    # python dictionaries from filename to set of labels.
    if len(sys.argv) < 2:
        raise ValueError("Please give an input filename/labels dict.")
    labels_dict_fname = sys.argv[1]

    assert os.path.isfile(labels_dict_fname)

    labels_dict = None

    with open(labels_dict_fname, 'rb') as labels_dict_file:
        labels_dict = pickle.load(labels_dict_file)

    fname_labels = labels_dict.values()
    shuffle(fname_labels)
    test_set = {}
    i = 0
    while i < test_size:
        test_set[fname_labels[i][0]] = fname_labels[i][1]
        i += 1
    valid_set = {}
    while i < test_size + valid_size:
        valid_set[fname_labels[i][0]] = fname_labels[i][1]
        i += 1
    train_set = {}
    while i < len(fname_labels):
        train_set[fname_labels[i][0]] = fname_labels[i][1]
        i += 1
    test_fname = 'test.pkl'
    valid_fname = 'valid.pkl'
    train_fname = 'train.pkl'

    with open(test_fname, 'w') as test_file:
        pickle.dump(test_set, test_file)
    with open(valid_fname, 'w') as valid_file:
        pickle.dump(valid_set, valid_file)
    with open(train_fname, 'w') as train_file:
        pickle.dump(train_set, train_file)
    

""" Makes a random test/validation/train split for the pixiv-115 dataset.
"""
import cPickle as pickle
import os
import os.path
import sys

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Please give an input tags file, split name, number of test samples and number of validation samples.")
    in_tags_fname = sys.argv[1]
    name = sys.argv[2]
    nb_test = int(sys.argv[3])
    nb_valid = int(sys.argv[4])
    in_tags = pickle.load(open(in_tags_fname, 'rb'))
    raw_tags = {}

    for folder in in_tags:
        for img_id in in_tags[folder]:
            

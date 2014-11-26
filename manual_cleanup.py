""" Scripts for manual checking of tags for a given dataset.
"""
import cPickle as pickle
import cv2
import sys
import os
import os.path
import numpy as np

from dataset import IdentityDataset

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Please give a tag pickle file, an image folder, and a suggestion model.")
    tags_fname = sys.argv[1]
    image_folder = sys.argv[2]
    sugg_model_fname = sys.argv[3]
    # Loads tags and suggestion model.
    tags = pickle.load(open(tags_fname, 'rb'))
    model = pickle.load(open(sugg_model_fname, 'rb'))
    for image_fname in tags:
        image = cv2.imread(os.path.join(image_folder, image_fname))
        cv2.imshow('image', image)
        cv2.waitKey(100)
        c01image = np.rollaxis(image.astype(np.float32) / 255., 2, 0)
        print "Ground truth:"
        for tag in tags[image_fname]:
            print tag
        print "Predicted:"
        predicted = model.mini_batch_predict_labels_named(
            IdentityDataset([c01image], [frozenset()]),
            batch_size=1,
            method=('thresh', 0.2)
        )
        for tag in predicted[0]:
            print tag
        cv2.waitKey(0)

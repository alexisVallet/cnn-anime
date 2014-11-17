import cv2
import numpy as np
import cPickle as pickle
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Please input a model file.")
    model_file = sys.argv[1]
    classifier = None
    print "Loading model..."
    with open(model_file, 'rb') as model:
        classifier = pickle.load(model)
    # Getting first layer filters.
    filters = classifier.model.conv_mp_layers[0].filters.get_value()
    nb_filters, nb_feat, nb_rows, nb_cols = filters.shape

    for i in range(nb_filters):
        filter_image = cv2.resize(
            np.rollaxis(filters[i], 0, 3),
            (96, 96),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow('filter ' + repr(i), filter_image)
        cv2.moveWindow('filter ' + repr(i), 128 + (i / 7) * 128, 128 + (i % 7) * 128)
    cv2.waitKey(0)

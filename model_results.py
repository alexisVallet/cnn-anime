""" Displays randomly selected images from a test set and the predicted labels according
    to a model.
"""
import cPickle as pickle
import sys
import numpy as np
import cv2

from dataset import load_pixiv_1M, LazyIO, IdentityDataset

if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise ValueError("Please input an output model file, image folder and test file.")
    model_file = sys.argv[1]
    image_folder = sys.argv[2]
    test_file = sys.argv[3]
    classifier = None
    print "Loading model..."
    with open(model_file, 'rb') as model:
        classifier = pickle.load(model)
    print "Loading test data..."
    test_data = load_pixiv_1M(
        image_folder,
        test_file,
        LazyIO
    )
    # shuffling
    test_data.shuffle(np.random.permutation(len(test_data)))
    # Predicting on the 50 first samples.
    samples = []
    labels = test_data.get_labels()[0:50]
    samples_iter = iter(test_data)

    for i in range(50):
        samples.append(samples_iter.next())
    test_subset = IdentityDataset(samples, labels)
    probas = classifier.predict_probas(test_subset)
    sorted_labels = np.argsort(probas, axis=1)
    int_to_label = classifier.named_conv.int_to_label

    for i in range(50):
        # Show the top 5 guesses.
        top5ints = sorted_labels[i,::-1][0:5]
        print "Predictions:"
        for j in range(5):
            print int_to_label[top5ints[j]] + ": " + repr(probas[i, top5ints[j]])
        print "Ground truth:"
        for l in labels[i]:
            print l
        cv2.imshow('image', np.rollaxis(samples[i], 0, 3))
        cv2.waitKey(0)

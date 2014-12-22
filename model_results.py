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
    int_to_label = classifier.named_conv.int_to_label

    for i, sample in enumerate(test_subset):
        # Run prediction
        probas = classifier.predict_probas(IdentityDataset([sample], labels[i]))
        sorted_labels = np.argsort(probas)
        # Show the top 5 guesses.
        top5ints = sorted_labels[0,::-1][0:5]
        print "Predictions:"
        for j in range(5):
            print int_to_label[top5ints[j]] + ": " + repr(probas[0,top5ints[j]])
        print "Ground truth:"
        for l in labels[i]:
            print l
        # Show confidence maps for top 5 guesses.
        maps = classifier.compute_activations(13, IdentityDataset([sample], labels[i]))
        max_activation = maps[0,top5ints].max()
        for j in range(5):
            map_image = cv2.GaussianBlur(maps[0,top5ints[j]], (5,5), 0)
            map_image /= max_activation
            map_image += 0.3
            map_image /= map_image.max()
            name = int_to_label[top5ints[j]].encode('utf-8')
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.imshow(name, map_image)
        cv2.imshow('image', np.rollaxis(samples[i], 0, 3))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

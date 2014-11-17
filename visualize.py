import cv2
import numpy as np
import cPickle as pickle
import sys
import theano

from dataset import load_pixiv_1M, LazyIO, IdentityDataset

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Please input an output model file, image folder, test file and a layer number to visualize.")
    model_file = sys.argv[1]
    image_folder = sys.argv[2]
    test_file = sys.argv[3]
    layer_number = int(sys.argv[4])
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
    # Predicting on (say) the 100 first samples.
    samples = []
    labels = test_data.get_labels()[0:50]
    samples_iter = iter(test_data)

    for i in range(50):
        samples.append(samples_iter.next())
    test_subset = IdentityDataset(samples, labels)

    # Compute activations, show the highest max-response feature maps for
    # each filter.
    activations = classifier.compute_activations(layer_number, test_subset)
    batch_size, nb_filters, rows, cols = activations.shape
    cv2.namedWindow('feature map', cv2.WINDOW_NORMAL)

    for i in range(nb_filters):
        print "Filter " + repr(i)
        for j in range(10):
            # Display the corresponding maps and images of randomly selected images.
            chosen_idx = np.random.randint(len(samples) * 10)
            fmap = activations[chosen_idx, i]
            cv2.imshow('feature map', fmap / fmap.max())
            cv2.imshow('image', np.rollaxis(samples[chosen_idx / 10], 0, 3))
            cv2.waitKey(0)

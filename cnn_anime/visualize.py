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
    test_subset.shuffle(np.random.permutation(len(test_subset)))

    # Compute activations, show the highest max-response feature maps for
    # each filter.

    for sample in test_subset:
        # Display the corresponding maps and images of randomly selected images.
        fmaps = classifier.compute_activations(
            layer_number,
            IdentityDataset([sample], [frozenset()])
        )
        print fmaps.shape
        # Look a the top 5:
        top5 = fmaps.mean(axis=2).mean(axis=2).flatten().argsort()[::-1][0:5]
        cv2.imshow('image', np.rollaxis(sample, 0, 3))
        for i in top5:
            fmap = fmaps[0,i]
            print repr(i) + ": " + classifier.named_conv.int_to_label[i]
            cv2.namedWindow('feature map ' + repr(i), cv2.WINDOW_NORMAL)
            cv2.imshow('feature map ' + repr(i), 1 - (fmap - fmap.min()) / (fmap.max() - fmap.min()))
            print (fmap.min(), fmap.mean(), fmap.max())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

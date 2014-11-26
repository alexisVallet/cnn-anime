""" Tests a model on test data.
"""

import cPickle as pickle
import sys

from dataset import load_pixiv_1M, LazyIO
from metrics import hamming_score, multi_label_recall, multi_label_precision, unscaled_recall, unscaled_precision, unscaled_accuracy

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
    print "Predicting..."
    accuracy, recall, precision = classifier.mlabel_metrics_named(
        test_data,
        25,
        metrics=[hamming_score, multi_label_recall, multi_label_precision],
        method=('thresh', 0.2)
    )
    print "Accuracy: " + repr(accuracy)
    print "Recall: " + repr(recall)
    print "Precision: " + repr(precision)

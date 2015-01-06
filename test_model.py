""" Tests a model on test data.
"""

import cPickle as pickle
import sys

from dataset import load_pixiv_1M, LazyIO
from metrics import hamming_score, multi_label_recall, multi_label_precision, unscaled_recall, unscaled_precision, unscaled_accuracy

if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise ValueError("Please input an output model file, image folder, validation file and test file.")
    model_file = sys.argv[1]
    image_folder = sys.argv[2]
    valid_file = sys.argv[3]
    test_file = sys.argv[4]
    classifier = None
    print "Loading model..."
    with open(model_file, 'rb') as model:
        classifier = pickle.load(model)
    classifier.preprocessing[1].prediction = 'full_size'
    print "Loading test data..."
    valid_data = load_pixiv_1M(
        image_folder,
        valid_file,
        LazyIO
    )
    test_data = load_pixiv_1M(
        image_folder,
        test_file,
        LazyIO
    )
    print "Predicting..."
    accuracy, recall, precision = classifier.spp_metrics_named(
        test_data,
        layer_number=13,
        method=('lin-thresh', valid_data, hamming_score),
        top1=True,
        pyramid=[3,5,7,9,11,13,15],
        metrics=[hamming_score, multi_label_recall, multi_label_precision]
    )
    pickle.dump(classifier, open(model_file + "_linthresh.pkl", 'wb'))
    print "Accuracy: " + repr(accuracy)
    print "Recall: " + repr(recall)
    print "Precision: " + repr(precision)
    

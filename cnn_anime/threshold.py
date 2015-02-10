""" Threshold learning for multi-label prediction.
"""
import numpy as np
import cPickle as pickle

from metrics import hamming_score, multi_label_precision
from dataset import load_pixiv_1M

def learn_threshold(confidences, labels, metric=hamming_score):
    """ Learns a threshold function from confidence values and ground truth
        labels.

    Arguments:
        confidences
            (nb_samples, nb_labels) shaped matrix containing confidence values
            for each sample and labels.
        labels
            list of frozensets of integer labels corresponding to columns in
            the confidences matrix.
        metric
            metric to maximize. Usually just accuracy.
    Returns:
        w, b where w is a (nb_labels,) shaped vector and b is a scalar, defining
        for confidence values x the threshold function t(x) = xw + b .
    """
    # First, compute target values for each sample. We pick the middle of the
    # two confidence points maximizing the metric.
    nb_samples, nb_labels = confidences.shape
    targets = np.empty([nb_samples])
    for i in range(nb_samples):
        sorted_labels = np.argsort(confidences[i])
        best_point = None
        best_value = None
        
        for j in range(nb_labels):
            value = metric([labels[i]], [frozenset(sorted_labels[j:])])
            if best_value == None or value > best_value:
                best_point = j
                best_value = value
        if best_point == 0:
            targets[i] = confidences[i, sorted_labels[best_point]]
        else:
            targets[i] = (
                confidences[i, sorted_labels[best_point - 1]] + confidences[i, sorted_labels[best_point]]
            ) / 2
    # Solving the corresponding linear problem.
    X = np.empty([nb_samples, nb_labels + 1])
    X[:,0:nb_labels] = confidences
    X[:,nb_labels] = 1
    wp = np.linalg.lstsq(X, targets)[0]
    w = wp[0:nb_labels]
    b = wp[nb_labels]
    
    return (w, b)

if __name__ == "__main__":
    confs = pickle.load(open('data/pixiv-115/gap-confidences.pkl', 'rb'))
    classifier = pickle.load(open('data/pixiv-115/models/deep-gap-padding/test_38_backup.pkl', 'rb'))
    valid_set = load_pixiv_1M('data/pixiv-115/images', 'data/pixiv-115/raw_valid.pkl')
    int_labels = map(lambda ls: frozenset(map(lambda l: classifier.named_conv.label_to_int[l], ls)), valid_set.get_labels())
    w, b = learn_threshold(confs, int_labels, multi_label_precision)
    

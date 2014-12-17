""" Metrics for accuracy, etc.
"""
import numpy as np

def multi_label_recall(expected, predicted):
    """ Multi-label accuracy, measures accuracy as a measure of how many samples
        were gotten right.

    Arguments:
        expected
            list of sets of integer labels. Ground truth labels.
        predicted
            list of sets of integer labels. Predicted labels.

    Returns:
        The multi-label recall.
    """
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    nb_correct = 0.
    
    for i in range(nb_samples):
        nb_correct += (float(len(expected[i].intersection(predicted[i])))
                        / len(expected[i]))
    return nb_correct / nb_samples

def multi_label_precision(expected, predicted):
    """ Measures multi-label precision.
    """
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    total = 0.

    for i in range(nb_samples):
        if len(predicted[i]) > 0:
            total += (float(len(expected[i].intersection(predicted[i])))
                      / len(predicted[i]))

    return total / nb_samples

def hamming_score(expected, predicted):
    """ Measures multi-label hamming score (also called accuracy in multi-label setting).
    """
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    total = 0.

    for i in range(nb_samples):
        total += (float(len(expected[i].intersection(predicted[i])))
                  / len(expected[i].union(predicted[i])))

    return total / nb_samples

def unscaled_recall(expected, predicted):
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    denom = 0.
    total = 0.

    for i in range(nb_samples):
        total += len(expected[i].intersection(predicted[i]))
        denom += len(expected[i])

    return total / denom

def unscaled_precision(expected, predicted):
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    denom = 0.
    total = 0.

    for i in range(nb_samples):
        total += len(expected[i].intersection(predicted[i]))
        denom += len(predicted[i])

    return total / denom

def unscaled_accuracy(expected, predicted):
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    denom = 0.
    total = 0.

    for i in range(nb_samples):
        total += len(expected[i].intersection(predicted[i]))
        denom += len(predicted[i].union(expected[i]))

    return total / denom

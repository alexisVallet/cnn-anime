""" Metrics for accuracy, etc.
"""
import numpy as np

def multi_label_sample_accuracy(expected, predicted):
    """ Multi-label accuracy, measures accuracy as a measure of how many samples
        were gotten right.

    Arguments:
        expected
            list of sets of integer labels. Ground truth labels.
        predicted
            list of sets of integer labels. Predicted labels.

    Returns:
        a measure of accuracy.
    """
    assert len(expected) == len(predicted)
    nb_samples = len(expected)
    nb_correct = 0.
    
    for i in range(nb_samples):
        nb_correct += (float(len(expected[i].intersection(predicted[i])))
                        / len(expected[i]))
    return nb_correct / nb_samples

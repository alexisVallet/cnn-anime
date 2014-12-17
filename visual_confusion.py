""" Generates a confusion matrix for a given character label by picking random true positives,
    false positives and false negatives out of a test set.
"""
import cPickle as pickle
import os.path
import sys
import random

from dataset import load_pixiv_1M, LazyIO

if __name__ == "__main__":
    if len(sys.argv) < 6:
        raise ValueError("Please input an images folder, test set, results file, label name and output folder.")
    img_folder = sys.argv[1]
    test_set = load_pixiv_1M(
        img_folder,
        sys.argv[2],
        LazyIO
    )
    results = pickle.load(open(sys.argv[3], 'rb'))
    char = sys.argv[4].decode('utf-8')
    out_folder = sys.argv[5]
    # Sort out all the TP, FP and FN
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    fnames = test_set.filenames
    ground_truth = test_set.get_labels()
    
    for i in range(len(test_set)):
        max_conf = -5000
        max_conf_label = None
        confidence = {}

        for label in results:
            confidence[label] = results[label][i]
            if results[label][i] > max_conf:
                max_conf = results[label][i]
                max_conf_label = label
        predicted = max_conf_label
        
        if char == predicted and char in ground_truth[i]:
            true_positives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
        elif char == predicted and char not in ground_truth[i]:
            false_positives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
        elif char != predicted and char in ground_truth[i]:
            false_negatives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
    # Save results
    pickle.dump(true_positives, open(os.path.join(out_folder, 'tp.pkl'), 'wb'))
    pickle.dump(false_positives, open(os.path.join(out_folder, 'fp.pkl'), 'wb'))
    pickle.dump(false_negatives, open(os.path.join(out_folder, 'fn.pkl'), 'wb'))

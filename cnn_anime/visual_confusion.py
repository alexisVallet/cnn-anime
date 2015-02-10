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
        raise ValueError("Please input an images folder, results file, and output folder.")
    img_folder = sys.argv[1]
    results = pickle.load(open(sys.argv[2], 'rb'))
    char = sys.argv[3].decode('utf-8')
    out_folder = sys.argv[4]
    # Sort out all the TP, FP and FN
    true_positives = {}
    false_positives = {}
    false_negatives = {}
    fnames = results.fnames
    ground_truth = results.ground_truth
    predicted = results.predicted
    nb_samples = len(fnames)
    
    for i in range(nb_samples):        
        if char in predicted[i] and char in ground_truth[i]:
            true_positives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
        elif char in predicted[i] and char not in ground_truth[i]:
            false_positives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
        elif char not in predicted[i] and char in ground_truth[i]:
            false_negatives[fnames[i]] = {
                'confidence': confidence,
                'ground_truth': ground_truth[i]
            }
    # Save results
    pickle.dump(true_positives, open(os.path.join(out_folder, 'tp.pkl'), 'wb'))
    pickle.dump(false_positives, open(os.path.join(out_folder, 'fp.pkl'), 'wb'))
    pickle.dump(false_negatives, open(os.path.join(out_folder, 'fn.pkl'), 'wb'))

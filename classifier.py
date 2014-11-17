""" Classifier mixin offering functionality including:
    - grid search
    - named labels
    - validation sets
    - accuracy scores
    - cross validation
"""
import numpy as np
import itertools
from random import shuffle

from metrics import hamming_score
from preprocessing import NameLabels
from dataset import mini_batch_split, IdentityDataset

class ClassifierMixin:    
    def train_named(self, train_samples, valid_samples=None):
        self.named_conv = NameLabels()
        int_train = self.named_conv.train_data_transform(train_samples)
        int_valid = None if valid_samples == None else self.named_conv.test_data_transform(valid_samples)
        self.train(int_train, int_valid)
        
    def predict_labels_named(self, test_samples):
        int_labels = self.predict_labels(
            self.named_conv.test_data_transform(
                test_samples
            )
        )
        
        return map(
            lambda ls: frozenset(map(lambda l: self.named_conv.int_to_label[l], ls)) if isinstance(ls, frozenset) else frozenset([self.named_conv.int_to_label[ls]]),
            int_labels
        )

    def mini_batch_predict_labels_named(self, test_samples, batch_size, method='top-1'):
        """ Run prediction by splitting the test data into mini-batches. Useful
            when the entire test set does not fit into GPU memory.
        """
        splits = mini_batch_split(test_samples, batch_size)
        nb_batches = splits.size - 1
        predicted_labels = []
        sample_it = iter(test_samples)
        test_labels = test_samples.get_labels()

        for i in range(nb_batches):
            print "batch " + repr(i)
            cur_batch_size = splits[i+1] - splits[i]
            batch = []
            batch_labels = test_labels[splits[i]:splits[i+1]]
            
            for j in range(cur_batch_size):
                batch.append(sample_it.next())
            int_labels = self.predict_labels(
                self.named_conv.test_data_transform(
                    IdentityDataset(batch, batch_labels)
                ),
                method=method
            )
            predicted_labels += map(
                lambda ls: frozenset(map(lambda l: self.named_conv.int_to_label[l], ls)) if isinstance(ls, frozenset) else frozenset([self.named_conv.int_to_label[ls]]),
                int_labels
            )
        return predicted_labels

    def top_accuracy(self, samples):
        """ Computes top-1 to to top-k accuracy of the classifier on test data,
            assuming it already has been trained, where k is the total number
            of classes.
        """
        probas = self.predict_proba(samples)
        sorted_classes = np.fliplr(np.argsort(probas, axis=1))
        nb_classes = probas.shape[1]
        nb_samples = len(samples)
        nb_correct_top = np.zeros([nb_classes], np.int32)
        sample_it = iter(samples)
        labels = samples.get_labels()

        for i in range(nb_samples):
            sample = sample_it.next()
            
            for j in range(nb_classes):
                if labels[i] == sorted_classes[i,j]:
                    nb_correct_top[j:] += 1
                    break

        return nb_correct_top.astype(np.float64) / nb_samples

    def mlabel_metrics_named(self, test_samples, batch_size=None, metrics=[hamming_score],
                             method='top-1'):
        """ Compute multi label accuracy, given a classifier that actually only
            outputs one (weird I know).
        """
        expected = map(
            lambda l: l if isinstance(l, frozenset) else frozenset([l]),
            test_samples.get_labels()
        )
        predicted = map(
            lambda l: l if isinstance(l, frozenset) else frozenset([l]),
            self.predict_labels_named(test_samples, method) if batch_size == None
            else self.mini_batch_predict_labels_named(test_samples, batch_size, method)
        )
        
        return map(lambda f: f(expected, predicted), metrics)

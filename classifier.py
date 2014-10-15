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

class ClassifierMixin:
    """ Mixin for classifiers adding grid search functionality, as well
        as named labels. The class should implement the following:
        - train(samples, int_labels)
        - predict(samples) -> int_labels
        - verbose -> boolean
        - a constructor which takes the arguments to perform grid search
          on, and fully resets the classifier (ie. one can call __init__
          multiple times without corrupting the state).
    """
    def predict_named(self, samples):
        int_labels = self.predict(samples)
        return map(
            lambda i: self.int_to_label[i],
            self.predict(samples).tolist()
        )

    def name_to_int(self, labels):
        """ Converts a collection of string labels to integer labels, storing
            the correspondance in the self.int_to_label list.
        """
        self.int_to_label = list(set(labels))
        self.label_to_int = {}
        
        for i in range(len(self.int_to_label)):
            self.label_to_int[self.int_to_label[i]] = i
        
        int_labels = np.array(
            map(lambda l: self.label_to_int[l], labels),
            dtype=np.int32
        )

        return int_labels

    def name_to_int_test(self, labels):
        return np.array(
            map(lambda l: self.label_to_int[l], labels),
            dtype=np.int32
        )

    def train_named(self, samples, labels):
        self.train(samples, self.name_to_int(labels))

    def train_gs_named(self, samples, labels, k, **args):
        """ Trains a classifier with grid search using named labels.
        """
        return self.train_gs(samples, self.name_to_int(labels), k, **args)
    
    def train_validation(self, samples, labels, valid_size=0.2):
        """ Trains the classifier, picking a random validation set out of the training
            data.

        Arguments:
            samples
                full training set of samples.
            labels
                labels for the training samples.
            valid_size
                fraction of the samples of each class in the dataset to pick. This is
                not simply picking a random subset of the samples, as we still would
                like each class to be represented equally - and by at least one sample.
        """
        nb_samples = len(samples)
        nb_classes = np.unique(labels).size
        assert nb_samples >= 2 * nb_classes
        # Group the samples per class.
        samples_per_class = []
        for i in range(nb_classes):
            samples_per_class.append([])
        for i in range(nb_samples):
            samples_per_class[labels[i]].append(samples[i])
        # For each class, split into training and validation sets.
        train_samples = []
        train_labels = []
        valid_samples = []
        valid_labels = []
        for i in range(nb_classes):
            # We need at least one validation sample and one training sample.
            nb_samples_class = len(samples_per_class[i])
            assert nb_samples_class >= 2
            nb_valid = min(
                nb_samples_class - 1,
                max(1, 
                    int(round(valid_size * nb_samples_class))
                )
            )
            nb_train = nb_samples_class - nb_valid
            # Pick the sets randomly.
            shflidxs = np.random.permutation(nb_samples_class)
            j = 0
            for k in range(nb_valid):
                valid_samples.append(samples_per_class[i][shflidxs[j]])
                valid_labels.append(i)
                j += 1
            for k in range(nb_train):
                train_samples.append(samples_per_class[i][shflidxs[j]])
                train_labels.append(i)
                j += 1
        # Run the actual training.
        self.train(train_samples, np.array(train_labels, np.int32),
                   valid_samples, np.array(valid_labels, np.int32))

    def train_validation_named(self, samples, labels, valid_size=0.2):
        self.train_validation(samples, self.name_to_int(labels), valid_size)

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

    def top_accuracy_named(self, samples, labels):
        return self.top_accuracy(samples, self.name_to_int_test(labels))

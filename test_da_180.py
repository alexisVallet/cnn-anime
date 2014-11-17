""" Tests for pre-training with pixiv-1M and fine-tuning on da-180.
"""
import unittest
import numpy as np
import cPickle as pickle
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dataset import ListDataset, load_da_180
from optimize import SGD
from cnn_classifier import CNNClassifier
from preprocessing import MeanSubtraction, RandomPatch, Resize, RandomFlip

class TestDA1000(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.samples, cls.labels = load_da_180('data/da-180')

    def test_da_180_finetune(self):
        # Performs leave-one-out cross-validation.
        nb_samples = len(self.samples)
        batch_size = 16
        nb_correct = 0

        for i in range(nb_samples):
            print "Loading pre-trained classifier..."
            p1m_classifier = None
            with open('data/pixiv-1M/models/pixiv_1M48.pkl', 'rb') as p1m_file:
                p1m_classifier = pickle.load(p1m_file)
            train_set = ListDataset(
                self.samples[0:i] + self.samples[i+1:],
                self.labels[0:i] + self.labels[i+1:]
            )
            test_set = ListDataset(
                [self.samples[i]],
                [self.labels[i]]
            )
            print "Initializing classifier for fold " + repr(i)
            print repr(len(train_set)) + " training samples"
            da1000_classifier = CNNClassifier(
                architecture=(
                    p1m_classifier.model.conv_mp_layers +
                    p1m_classifier.model.fc_layers +
                    [('softmax', {'nb_outputs': 12})]
                ),
                optimizer=SGD(
                    batch_size=batch_size,
                    init_rate=0.01,
                    nb_epochs=30,
                    learning_schedule='constant',
                    update_rule=('rmsprop', 0.9, 0.01),
                    verbose=1
                ),
                srng=RandomStreams(seed=158907691),
                l2_reg = 0.0005,
                input_shape=[3,224,224],
                init='random',
                preprocessing=[
                    Resize(256),
                    MeanSubtraction(3),
                    RandomPatch(3, 224, 224, 10),
                    RandomFlip()
                ],
                verbose=True
            )
            print "Training..."
            da1000_classifier.train_named(train_set)
            print "Predicting..."
            label = da1000_classifier.predict_labels_named(test_set)
            # multi-label on paper so awkward.
            for l1 in label[0]:
                for l2 in self.labels[i]:
                    print "expected: " + repr(l2)
                    print "actual: " + repr(l1)
                    if l1 == l2:
                        nb_correct += 1
            print "Current LOO accuracy: " + repr(float(nb_correct) / (i + 1))
        print "LOO accuracy: " + repr(float(nb_correct) / nb_samples)

if __name__ == "__main__":
    unittest.main()

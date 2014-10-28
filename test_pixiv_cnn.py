""" Unit tests for the convnet classifier using the pixiv-!M dataset.
"""
import unittest
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cnn_classifier import CNNClassifier
from dataset import load_pixiv_1M, CompressedDataset
from preprocessing import MeanSubtraction, SingleLabelConversion, NameLabels, FixedPatches
from optimize import SGD

class TestPixiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.train_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/small_train.pkl',
            CompressedDataset
        )
        cls.valid_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/small_valid.pkl',
            CompressedDataset
        )
        cls.test_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/small_test.pkl',
            CompressedDataset
        )

    def test_pixiv(self):
        print "Initializing classifier..."

        batch_size = 256
        classifier = CNNClassifier(
            architecture=[
                ('conv', 24, 7, 7, 2, 2),
                ('max-pool', 2),
                ('conv', 64, 5, 5, 1, 1),
                ('max-pool', 2),
                ('conv', 98, 5, 5, 1, 1),
                ('conv', 98, 5, 5, 1, 1),
                ('conv', 64, 3, 3, 1, 1),
                ('max-pool', 2),
                ('fc', 512),
                ('softmax', 100)
            ],
            optimizer=SGD(
                batch_size=batch_size,
                init_rate=0.001,
                nb_epochs=10,
                learning_schedule=('decay', 0.9, 300),
                update_rule=('rmsprop', 0.9, 0.01),
                verbose=2
            ),
            srng=RandomStreams(seed=156736127),
            l2_reg = 0,
            input_shape=[3,224,224],
            init='random',
            preprocessing=[
                FixedPatches(3, 224),
                MeanSubtraction(3)
            ],
            verbose=True
        )
        print "Training..."
        classifier.train_named(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.test_data, batch_size)

if __name__ == "__main__":
    unittest.main()

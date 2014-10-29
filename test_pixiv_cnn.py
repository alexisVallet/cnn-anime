""" Unit tests for the convnet classifier using the pixiv-!M dataset.
"""
import unittest
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cnn_classifier import CNNClassifier
from dataset import load_pixiv_1M, CompressedDataset, LazyIO
from preprocessing import MeanSubtraction, SingleLabelConversion, NameLabels, FixedPatches
from optimize import SGD

class TestPixiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.train_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/small_train.pkl',
            LazyIO
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

        batch_size = 128
        classifier = CNNClassifier(
            architecture=[
                ('conv', {'nb_filters': 48, 'rows': 7, 'cols': 7, 'stride_r': 2, 'stride_c': 2}),
                ('max-pool', 2),
                ('conv', {'nb_filters': 128, 'rows': 5, 'cols': 5, 'init_bias': 1.}),
                ('max-pool', 2),
                ('conv', {'nb_filters': 192, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
                ('conv', {'nb_filters': 192, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
                ('max-pool', 2),
                ('fc', {'nb_units': 2048, 'init_bias': 1.}),
                ('softmax', {'nb_outputs': 100})
            ],
            optimizer=SGD(
                batch_size=batch_size,
                init_rate=0.01,
                nb_epochs=50,
                learning_schedule=('decay', 0.9, 300),
                update_rule=('rmsprop', 0.9, 0.01),
                verbose=2
            ),
            srng=RandomStreams(seed=156736127),
            l2_reg = 0,
            input_shape=[3,224,224],
            init='random',
            preprocessing=[
                FixedPatches(3, 224)
            ],
            verbose=True
        )
        print "Training..."
        classifier.train_named(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.test_data, batch_size)

if __name__ == "__main__":
    unittest.main()

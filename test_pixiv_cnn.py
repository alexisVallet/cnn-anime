""" Unit tests for the convnet classifier using the pixiv-!M dataset.
"""
import unittest
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle

from cnn_classifier import CNNClassifier
from dataset import load_pixiv_1M, CompressedDataset, LazyIO
from preprocessing import MeanSubtraction, NameLabels, RandomPatch, Resize, RandomFlip
from optimize import SGD

class TestPixiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.train_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/1M_train.pkl',
            LazyIO
        )
        cls.valid_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/1M_valid.pkl',
            LazyIO
        )
        cls.test_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/1M_test.pkl',
            LazyIO
        )

    def test_pixiv(self):
        print "Initializing classifier..."

        batch_size = 128
        # classifier = CNNClassifier(
        #     architecture=[
        #         ('conv', {'nb_filters': 48, 'rows': 7, 'cols': 7, 'stride_r': 2, 'stride_c': 2}),
        #         ('max-pool', 2),
        #         ('conv', {'nb_filters': 128, 'rows': 5, 'cols': 5, 'init_bias': 1.}),
        #         ('max-pool', 2),
        #         ('conv', {'nb_filters': 192, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
        #         ('conv', {'nb_filters': 192, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.}),
        #         ('max-pool', 2),
        #         ('fc', {'nb_units': 2048, 'init_bias': 1.}),
        #         ('dropout', 0.5),
        #         ('softmax', {'nb_outputs': 100})
        #     ],
        #     optimizer=SGD(
        #         batch_size=batch_size,
        #         init_rate=0.01,
        #         nb_epochs=50,
        #         learning_schedule=('decay', 0.9, 300),
        #         update_rule=('rmsprop', 0.9, 0.01),
        #         pickle_schedule=(1, 'data/pixiv-1M/models/pixiv_1M'),
        #         verbose=2
        #     ),
        #     srng=RandomStreams(seed=156736127),
        #     l2_reg = 0.0005,
        #     input_shape=[3,224,224],
        #     init='random',
        #     preprocessing=[
        #         Resize(256),
        #         MeanSubtraction(3),
        #         RandomPatch(3, 224, 224, 10),
        #         RandomFlip()
        #     ],
        #     verbose=True
        # )
        with open('data/pixiv-1M/models/pixiv_1M44_1.pkl', 'rb') as classifier_file:
            classifier = pickle.load(classifier_file)
        print "Training..."
        classifier.train_named(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.test_data, batch_size)

if __name__ == "__main__":
    unittest.main()

""" Unit tests for the convnet classifier using the pixiv-!M dataset.
"""
import unittest
import theano
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle

from cnn_classifier import CNNClassifier
from dataset import load_pixiv_1M, CompressedDataset, LazyIO, ListDataset
from preprocessing import MeanSubtraction, NameLabels, RandomPatch, Resize, RandomFlip
from optimize import SGD

class TestPixiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.train_data = load_pixiv_1M(
            'data/pixiv-115/images',
            'data/pixiv-115/raw_train.pkl',
            LazyIO
        )
        cls.valid_data = load_pixiv_1M(
            'data/pixiv-115/images',
            'data/pixiv-115/raw_valid.pkl',
            LazyIO
        )
        cls.test_data = load_pixiv_1M(
            'data/pixiv-115/images',
            'data/pixiv-115/raw_test.pkl',
            LazyIO
        )

    def test_pixiv(self):
        print "Initializing classifier..."

        batch_size = 128
        classifier = CNNClassifier(
            architecture=[
                ('conv', {'nb_filters': 64, 'rows': 7, 'cols': 7, 'stride_r': 2, 'stride_c': 2,
                          'padding': (3,3)}),
                ('max-pool', {'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2}),                
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'stride_r': 2, 'stride_c': 2, 'padding': (1,1)}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('max-pool', {'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2}),
                ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'init_bias': 1.,
                          'padding': (1,1)}),
                ('conv', {'nb_filters': 115, 'rows': 1, 'cols': 1, 'init_bias': 1.,
                          'padding': (1,1)}),
                'avg-pool'
            ],
            optimizer=SGD(
                batch_size=batch_size,
                init_rate=0.01,
                nb_epochs=70,
                learning_schedule=('decay', 0.9, 300),
                update_rule=('rmsprop', 0.9, 0.01),
                pickle_schedule=(1, 'data/pixiv-115/models/gap-orth/test'),
                verbose=2
            ),
            srng=RandomStreams(seed=156736127),
            l2_reg = 0,
            orth_penalty = 0.02,
            input_shape=[3,224,224],
            init='orth',
            cost='mlr',
            preprocessing=[
                MeanSubtraction(3, 'data/pixiv-115/raw_mean_pixel.pkl'),
                RandomPatch(3, 224, 224, ('rand_subwin', 10)),
                RandomFlip()
            ],
            verbose=True
        )
        print "Training..."
        classifier.train_named(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.test_data, batch_size)

if __name__ == "__main__":
    unittest.main()

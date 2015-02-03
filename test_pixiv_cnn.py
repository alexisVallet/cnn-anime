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

        # batch_size = 96
        # classifier = CNNClassifier(
        #     architecture=[
        #         ('conv', {'nb_filters': 64, 'rows': 7, 'cols': 7, 'padding': (3, 3),
        #                   'stride_r': 2, 'stride_c': 2}),
        #         ('pool', {'type': 'max', 'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'stride_r': 2, 'stride_c': 2, 'init_bias': 1.}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('pool', {'type': 'max', 'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2}),
        #         ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 256, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('pool', {'type': 'max', 'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2}),
        #         ('conv', {'nb_filters': 512, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 512, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 512, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 512, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('conv', {'nb_filters': 512, 'rows': 3, 'cols': 3, 'padding': (1, 1),
        #                   'init_bias': 1.}),
        #         ('pool', {'type': 'average', 'rows': 7, 'cols': 7, 'stride_r': 1, 'stride_c': 1}),
        #         ('conv', {'nb_filters': 4096, 'rows': 1, 'cols': 1, 'init_bias': 1.}),
        #         ('dropout', 0.5),
        #         ('conv', {'nb_units': 4096, 'init_bias': 1.}),
        #         ('dropout', 0.5),
        #         ('linear', {'nb_outputs': 115})
        #     ],
        #     optimizer=SGD(
        #         batch_size=batch_size,
        #         init_rate=0.001,
        #         nb_epochs=70,
        #         learning_schedule=('decay', 0.9, 300),
        #         update_rule=('rmsprop', 0.9, 0.01),
        #         pickle_schedule=(1, 'data/pixiv-115/models/gap15/gap15_dropout'),
        #         verbose=2
        #     ),
        #     srng=RandomStreams(seed=156736127),
        #     l2_reg = 0,
        #     input_shape=[3,256,256],
        #     init='random',
        #     cost='mlr',
        #     preprocessing=[
        #         MeanSubtraction(3, 'data/pixiv-115/raw_mean_pixel.pkl'),
        #         RandomPatch(3, 256, 256, ('rand_subwin', 10)),
        #         RandomFlip()
        #     ],
        #     verbose=True
        # )
        classifier = pickle.load(open('data/pixiv-115/models/gap15/gap15_dropout_50_backup.pkl', 'rb'))
        classifier.architecture = classifier.architecture[0:20] + [
            ('fc', {'nb_units': 4096, 'init_bias': 1.}),
            ('dropout', 0.5),
            ('fc', {'nb_units': 4096, 'init_bias': 1.}),
            ('dropout', 0.5),
            ('fc', {'nb_units': 115})
        ]
        classifier.init_model()
        classifier.optimizer.batch_size = 128
        print "Training..."
        classifier.train_named(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.test_data, batch_size)

if __name__ == "__main__":
    unittest.main()

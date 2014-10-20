""" Unit tests for the convnet classifier using the pixiv-!M dataset.
"""
import unittest
import theano
import numpy as np

from cnn_classifier import CNNClassifier
from dataset import load_pixiv_1M
from preprocessing import MeanSubtraction, SingleLabelConversion, NameLabels, FixedPatches
from optimize import SGD

class TestPixiv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.train_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/train.pkl'
        )
        cls.valid_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/valid.pkl'
        )
        cls.test_data = load_pixiv_1M(
            'data/pixiv-1M/images',
            'data/pixiv-1M/test.pkl'
        )

    def test_pixiv(self):
        print "Initializing classifier..."
        
        classifier = CNNClassifier(
            architecture=[
                ('conv', 48, 11, 11, 4, 4),
                ('max-pool', 2),
                ('conv', 128, 5, 5, 1, 1),
                ('max-pool', 2),
                ('conv', 192, 3, 3, 1, 1),
                ('conv', 192, 3, 3, 1, 1),
                ('conv', 128, 3, 3, 1, 1),
                ('max-pool', 2),
                ('fc', 2048),
                ('fc', 2048),
                ('softmax', 100)
            ],
            optimizer=SGD(
                batch_size=256,
                init_rate=0.001,
                nb_epochs=15,
                learning_schedule=('decay', 0.1, 5),
                update_rule=('momentum', 0.9),
                verbose=2
            ),
            l2_reg = 0,
            input_shape=[3,256,256],
            init='random',
            preprocessing=[
                SingleLabelConversion(),
                NameLabels(),
                FixedPatches(3, 256)
            ],
            verbose=True
        )
        print "Training..."
        classifier.train(self.train_data, self.valid_data)
        print "Predicting..."
        print classifier.top_accuracy(self.testdata)[0:10]

if __name__ == "__main__":
    unittest.main()

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
                ('conv', 64, 5, 5, 'cuda-convnet'),
                ('max-pool', 4),
                ('softmax', 1024)
            ],
            optimizer=SGD(
                batch_size=256,
                init_rate=0.001,
                nb_epochs=1,
                learning_schedule=('decay', 0.1),
                update_rule=('momentum', 0.9),
                verbose=2
            ),
            l2_reg = 5E-4,
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

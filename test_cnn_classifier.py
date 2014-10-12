""" Unit tests for the convnet classifier.
"""
import unittest
import numpy as np
import cv2

from cnn_classifier import CNNClassifier
from dataset import load_mnist
from optimize import GD, SGD

class TestCNNClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        # Loads the MNIST dataset.
        cls.trainsamples, cls.trainlabels = load_mnist(
            'data/mnist/train-images.idx3-ubyte',
            'data/mnist/train-labels.idx1-ubyte'
        )
        cls.testsamples, cls.testlabels = load_mnist(
            'data/mnist/t10k-images.idx3-ubyte',
            'data/mnist/t10k-labels.idx1-ubyte'
        )

    def test_cnn_classifier(self):
        print "Initializing classifier..."
        classifier = CNNClassifier(
            architecture=[
                ('conv', 8, 5, 5),
                ('max-pool', 4),
                ('fc', 256),
                ('softmax', 10)
            ],
            optimizer=SGD(
                batch_size=20,
                init_rate=0.001,
                nb_epochs=10,
                learning_schedule='fixed',
                update_rule=('momentum', 0.9),
                verbose=True
            ),
            input_shape=[1,28,28],
            init='random',
        )
        print "Training..."
        classifier.train(self.trainsamples, self.trainlabels)
        print "Predicting..."
        print classifier.top_accuracy(self.testsamples, self.testlabels)

if __name__ == "__main__":
    unittest.main()

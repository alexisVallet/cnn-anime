""" Unit tests for the convnet classifier.
"""
import unittest
import numpy as np
import cv2

from cnn_classifier import CNNClassifier
from dataset import load_mnist
from optimize import GD

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
        # Tests the CNN classifier with a simple LeNet-like
        # architecture to classify digits from the MNIST dataset.
        print "Initializing classifier..."
        classifier = CNNClassifier(
            architecture=[
                ('conv', 6, 5, 5), # 28 by 28 to 24 by 24
                ('max-pool', 2), # to 12 by 12
                ('conv', 16, 5, 5), # to 8 by 8
                ('max-pool', 2), # to 4 by 4
                ('conv', 120, 4, 4), # to 1 by 1
                ('fc', 120, 84), # Each 1 by 1 to a 84-neuron net
                ('softmax', 84, 10) # 10 classes, one for each digit.
            ],
            optimizer=GD(
                learning_rate=0.001,
                nb_iter=100,
                nb_samples=len(self.trainsamples),
                verbose=True
            ),
            init='random',
            nb_channels=1
        )
        print "Training..."
        classifier.train(self.trainsamples, self.trainlabels)
        print "Predicting..."
        print classifier.top_accuracy(self.testsamples, self.testlabels)

if __name__ == "__main__":
    unittest.main()

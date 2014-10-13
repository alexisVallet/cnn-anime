""" Unit tests for the convnet classifier.
"""
import unittest
import numpy as np
import cv2

from cnn_classifier import CNNClassifier
from dataset import load_mnist, ListDataset
from optimize import SGD

class TestCNNClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        # Loads the MNIST dataset.
        trainsamples, trainlabels = load_mnist(
            'data/mnist/train-images.idx3-ubyte',
            'data/mnist/train-labels.idx1-ubyte'
        )
        # Choose a validation dataset.
        validation_size = 5000
        perm = np.random.permutation(len(trainsamples))
        trainsamples.shuffle(perm)
        validsamples = []
        trainsamples_ = []
        cls.validlabels = trainlabels[perm[0:validation_size]]
        cls.trainlabels = trainlabels[perm[validation_size:]]
        i = 0

        for sample in trainsamples:
            if i < validation_size:
                validsamples.append(sample)
            else:
                trainsamples_.append(sample)
            i += 1
        cls.trainsamples = ListDataset(trainsamples_)
        cls.validsamples = ListDataset(validsamples)
        
        cls.testsamples, cls.testlabels = load_mnist(
            'data/mnist/t10k-images.idx3-ubyte',
            'data/mnist/t10k-labels.idx1-ubyte'
        )

    def test_cnn_classifier(self):
        print "Initializing classifier..."
        classifier = CNNClassifier(
            architecture=[
                ('conv', 8, 5, 5),
                ('max-pool', 2),
                ('fc', 512),
                ('softmax', 10)
            ],
            optimizer=SGD(
                batch_size=32,
                init_rate=0.001,
                nb_epochs=10,
                learning_schedule=(
                    'decay', 
                    0.1, 
                    self.validsamples,
                    self.validlabels
                ),
                update_rule=('momentum', 0.9),
                verbose=True
            ),
            l2_reg=10E-3,
            input_shape=[1,28,28],
            init='random',
            verbose=True
        )
        print "Training..."
        classifier.train(self.trainsamples, self.trainlabels)
        print "Predicting..."
        print classifier.top_accuracy(self.testsamples, self.testlabels)

if __name__ == "__main__":
    unittest.main()

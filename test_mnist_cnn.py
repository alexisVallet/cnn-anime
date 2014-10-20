""" Unit tests for the convnet classifier using the MNIST dataset.
"""
import unittest
import theano
import numpy as np
import cv2

from cnn_classifier import CNNClassifier
from dataset import load_mnist, ListDataset
from preprocessing import MeanSubtraction
from optimize import SGD

class TestMNIST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        # Loads the MNIST dataset.
        traindata = load_mnist(
            'data/mnist/train-images.idx3-ubyte',
            'data/mnist/train-labels.idx1-ubyte'
        )
        # Choose a validation dataset.
        validation_size = 5000
        perm = np.random.permutation(len(traindata))
        traindata.shuffle(perm)
        validsamples = []
        trainsamples = []
        trainlabels = traindata.get_labels()
        validlabels = trainlabels[0:validation_size]
        trainlabels = trainlabels[validation_size:]
        i = 0

        for sample in traindata:
            if i < validation_size:
                validsamples.append(sample)
            else:
                trainsamples.append(sample)
            i += 1
        cls.traindata = ListDataset(trainsamples, trainlabels)
        cls.validdata = ListDataset(validsamples, validlabels)
        
        cls.testdata = load_mnist(
            'data/mnist/t10k-images.idx3-ubyte',
            'data/mnist/t10k-labels.idx1-ubyte'
        )

    def test_cnn_classifier(self):
        print "Initializing classifier..."
        
        classifier = CNNClassifier(
            architecture=[
                ('conv', 8, 5, 5, 1, 1),
                ('max-pool', 2),
                ('conv', 16, 5, 5, 1, 1),
                ('max-pool', 2),
                ('conv', 128, 3, 3, 1, 1),
                ('fc', 512),
                ('softmax', 10)
            ],
            optimizer=SGD(
                batch_size=32,
                init_rate=0.001,
                nb_epochs=15,
                learning_schedule=('decay', 0.1, 5),
                update_rule=('momentum', 0.9),
                verbose=1
            ),
            l2_reg=10E-3,
            input_shape=[1,28,28],
            init='random',
            preprocessing=[MeanSubtraction(1)],
            verbose=True
        )
        print "Training..."
        classifier.train(self.traindata, self.validdata)
        print "Predicting..."
        print classifier.top_accuracy(self.testdata)

if __name__ == "__main__":
    unittest.main()

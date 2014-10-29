""" Unit tests for the convnet classifier using the MNIST dataset.
"""
import unittest
import theano
import numpy as np
import cv2
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from cnn_classifier import CNNClassifier
from dataset import load_mnist, ListDataset
from preprocessing import MeanSubtraction, NameLabels
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

        batch_size = 64
        classifier = CNNClassifier(
            architecture=[
                ('conv', {
                    'nb_filters': 4,
                    'rows': 3,
                    'cols': 3,
                    'init_bias': 1.
                }),
                ('max-pool', 2),
                ('conv', {
                    'nb_filters': 16,
                    'rows': 3,
                    'cols': 3,
                    'init_bias': 1.
                }),
                ('max-pool', 2),
                ('conv', {
                    'nb_filters': 8,
                    'rows': 1,
                    'cols': 1,
                    'init_bias': 1.
                }),
                ('conv', {
                    'nb_filters': 128,
                    'rows': 3,
                    'cols': 3,
                    'init_bias': 1.
                }),
                'avg-pool',
                ('softmax', {
                    'nb_outputs': 10
                })
            ],
            optimizer=SGD(
                batch_size=batch_size,
                init_rate=0.001,
                nb_epochs=50,
                learning_schedule=('decay', 0.9, 10),
                update_rule=('momentum', 0.9),
                verbose=1
            ),
            srng=RandomStreams(seed=156736127),
            l2_reg=0,
            input_shape=[1,28,28],
            init='random',
            preprocessing=[
                MeanSubtraction(1),
            ],
            verbose=True
        )
        print "Training..."
        classifier.train_named(self.traindata, self.validdata)
        print "Predicting..."
        print classifier.mlabel_accuracy_named(self.testdata, batch_size)

if __name__ == "__main__":
    unittest.main()

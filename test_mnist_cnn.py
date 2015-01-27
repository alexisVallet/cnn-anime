""" Unit tests for the convnet classifier using the MNIST dataset.
"""
import unittest
import theano
import numpy as np
import cv2
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pickle

from cnn_classifier import CNNClassifier
from dataset import load_mnist, ListDataset, IdentityDataset
from preprocessing import MeanSubtraction, NameLabels, RandomPatch
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
        cls.traindata = IdentityDataset(trainsamples, trainlabels)
        cls.validdata = IdentityDataset(validsamples, validlabels)
        
        cls.testdata = load_mnist(
            'data/mnist/t10k-images.idx3-ubyte',
            'data/mnist/t10k-labels.idx1-ubyte'
        )

    def test_cnn_classifier(self):
        print "Initializing classifier..."
        preprocessing = [
            MeanSubtraction(1),
            RandomPatch(1, 28, 28, ('random_subwin', 10))
        ]
        batch_size = 128
        classifier = CNNClassifier(
            architecture=[
                ('conv', {'nb_filters': 32, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 32, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('max-pool', {'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2.}),
                ('conv', {'nb_filters': 64, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 64, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 64, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('max-pool', {'rows': 3, 'cols': 3, 'stride_r': 2, 'stride_c': 2.}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 128, 'rows': 3, 'cols': 3,
                          'padding': (1,1), 'non_lin': None}),
                ('conv', {'nb_filters': 10, 'rows': 3, 'cols': 3, 'padding': (1,1),
                          'non_lin': None}),
                'avg-pool'
            ],
            optimizer=SGD(
                batch_size=batch_size,
                init_rate=0.001,
                nb_epochs=10,
                learning_schedule=('decay', 0.9, 10),
                update_rule=('rmsprop', 0.9, 0.01),
                pickle_schedule=(10, 'data/mnist/models/test_model'),
                verbose=2
            ),
            srng=RandomStreams(seed=156736127),
            l2_reg=0,
            orth_penalty=0,
            input_shape=[1,28,28],
            init='random',
            cost='mlr',
            preprocessing=preprocessing,
            verbose=True
        )
        print "Training..."
        classifier.train_named(self.traindata, self.validdata)
        print "Predicting..."
        print classifier.mlabel_metrics_named(self.testdata, batch_size)
        print "Saving model..."
        model_fname = 'test_mnist.pkl'
        with open(model_fname, 'wb') as outfile:
            pickle.dump(
                classifier,
                outfile,
                protocol=pickle.HIGHEST_PROTOCOL
            )
        print "Reloading and testing..."
        with open(model_fname, 'rb') as infile:
            loaded_classifier = pickle.load(
                infile
            )
            print loaded_classifier.mlabel_metrics_named(self.testdata, batch_size)

if __name__ == "__main__":
    unittest.main()

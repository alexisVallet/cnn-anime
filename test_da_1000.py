""" Tests for pre-training with pixiv-1M and fine-tuning on da-1000.
"""
import unittest
import numpy as np
import cPickle as pickle
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from dataset import ListDataset, load_da_1000
from optimize import SGD
from cnn_classifier import CNNClassifier, NoTraining
from preprocessing import MeanSubtraction, RandomPatch, Resize, RandomFlip
from metrics import hamming_score

class TestDA1000(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print "Loading data..."
        cls.folds_img, cls.folds_lbl = load_da_1000('data/da-1000', bb_folder='data/da-1000/bounding_boxes/')

    def test_da_1000_finetune(self):
        # Make a new classifier by just reinitializing the softmax:
        nb_folds = len(self.folds_img)
        batch_size = 32
        for i in range(nb_folds):
            print "Loading pre-trained classifier..."
            p1m_classifier = None
            with open('data/pixiv-115/models/mlr-l2/mlr_l2_30.pkl', 'rb') as p1m_file:
                p1m_classifier = pickle.load(p1m_file)
            train_set = ListDataset(
                reduce(lambda l1, l2: l1 + l2, self.folds_img[0:i] + self.folds_img[i+1:]),
                reduce(lambda l1, l2: l1 + l2, self.folds_lbl[0:i] + self.folds_lbl[i+1:])
            )
            test_set = ListDataset(
                self.folds_img[i],
                self.folds_lbl[i]
            )
            print "Initializing classifier for fold " + repr(i)
            print repr(len(train_set)) + " training samples"
            da1000_classifier = CNNClassifier(
                architecture=(
                    map(NoTraining, p1m_classifier.model.layers[0:10]) +
                    [('linear', {'nb_outputs': 20})]
                ),
                optimizer=SGD(
                    batch_size=batch_size,
                    init_rate=0.01,
                    nb_epochs=30,
                    learning_schedule='constant',
                    update_rule=('rmsprop', 0.9, 0.01),
                    verbose=1
                ),
                srng=RandomStreams(seed=158907691),
                l2_reg = 0.0005,
                input_shape=[3,224,224],
                init='random',
                preprocessing=[
                    Resize(256),
                    MeanSubtraction(3),
                    RandomPatch(3, 224, 224, 10),
                    RandomFlip()
                ],
                verbose=True
            )
            print "Training..."
            da1000_classifier.train_named(train_set)
            print "Accuracy: " + repr(da1000_classifier.mlabel_metrics_named(
                test_set,
                30,
                metrics=[hamming_score],
                method='top-1'
            ))
    
if __name__ == "__main__":
    unittest.main()

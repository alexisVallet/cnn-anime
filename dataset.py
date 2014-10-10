import cv2
import numpy as np

class Dataset:
    """ A dataset of samples intended for comsumption by a convnet. 3 rules:
        - a dataset is an iterator of samples (method next())
        - there is a finite, nonzero and known number of samples (len(dataset))
        - all samples are numpy tensors with known and constant shape (list attribute sample_shape)
    """
    def next(self):
        """ A dataset of sample is an iterator of samples.
        """
        raise NotImplementedError()

    def __len__(self):
        """ Returns the number of samples of the dataset.
        """
        raise NotImplementedError()

class DatasetMixin:
    def to_array(self):
        """ Dumps the whole dataset into a big numpy array. Do not use
            unless you know the dataset fits into memory.

        Returns:
           a (len(self) + self.sample_shape) shaped numpy array, containing
           all the samples stacked together.
        """
        samples_array = np.empty([len(self)] + self.sample_shape)
        i = 0
        for sample in self:
            samples_array[i] = sample
            i += 1

        return samples_array

class BaseListDataset(Dataset):
    """ A very simple dataset implemented as a list of samples.
    """
    def __init__(self, samples):
        assert len(samples) > 0
        self.samples = samples
        self.sample_shape = list(self.samples[0].shape)

    def next(self):
        return self.samples.next()

    def __len__(self):
        return len(self.samples)

class ListDataset(BaseListDataset, DatasetMixin):
    pass

def load_mnist(mnist_pkl_filename):
    """ Load the MNIST dataset in a list dataset from a pickled file.

    Arguments:
        mnist_pkl_filename
            filename of the pickled mnist dataset.
    Returns:
        trainsamples, t
    """

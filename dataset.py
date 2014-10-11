import cv2
import theano
import numpy as np
import os, struct
from array import array

class Dataset:
    """ A dataset of samples intended for comsumption by a convnet. 3 rules:
        - a dataset is an iterator of samples (method next())
        - there is a finite, nonzero and known number of samples (len(dataset))
        - all samples are numpy tensors with known and constant shape (list attribute sample_shape)
    """
    def __iter__(self):
        raise NotImplementedError()

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
        samples_array = np.empty(
            [len(self)] + self.sample_shape,
            theano.config.floatX
        )
        i = 0
        for sample in iter(self):
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
        assert np.all([list(sample.shape) == self.sample_shape for sample in samples])

    def __iter__(self):
        self.cur_idx = 0

        return self

    def next(self):
        if self.cur_idx >= len(self.samples):
            raise StopIteration
        self.cur_idx += 1

        return self.samples[self.cur_idx - 1]

    def __len__(self):
        return len(self.samples)

class ListDataset(BaseListDataset, DatasetMixin):
    pass

def load_mnist(img_fname, lbl_fname):
    """ Load the MNIST dataset in a list dataset from a pickled file.
        This code was heavily inspired by cvxopt's (GPL licensed) code 
        for loading the MNIST dataset.

    Arguments:
        img_fname
            filename of the mnist image data to load in ubyte format (as
            provided on http://yann.lecun.com/exdb/mnist/ after
            extracting the .gz archive).
        lbl_fname
            filename of the corresponding mnist label data in ubyte
            format.
    Returns:
        images, labels where images is a ListDataset and labels a
        numpy array of integer labels, where the label number corresponds
        to the digit.
    """
    # Loading the raw label data.
    flbl = open(lbl_fname, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()
    
    # Loading the raw image data.
    fimg = open(img_fname, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()

    # Putting it all into the proper format.
    images = []
    labels = np.empty([size], dtype=np.int32)

    for i in range(size):
        # Convert the image to floating point [0;1] range while we're at it.
        image = np.array(
            img[i*rows*cols:(i+1)*rows*cols],
            theano.config.floatX
        ).reshape([1, rows, cols]) / 255.
        images.append(image)
        labels[i] = lbl[i]

    return ListDataset(images), labels

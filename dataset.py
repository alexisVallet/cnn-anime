import cv2
import theano
import numpy as np
import os, struct
from array import array
import cPickle as pickle

def load_pixiv_1M(images_folder, set_pkl):
    """ Loads a dataset in the pixiv-1M format as a LazyIO dataset with multi-labels.

    Arguments:
        images_folder
            folder where all the images are actually contained.
        set_pkl
            pickle file specifying the actual set (e.g. test, training, validation).
    Returns:
        A LazyIO dataset of images with multilabels. As the dataset is typically much
        too large to fit in memory, the images are not actually loaded here and everything is
        done on the fly.
    """
    fname_to_labels = None
    with open(set_pkl, 'rb') as dataset_file:
        fname_to_label = pickle.load(dataset_file)
    filenames = []
    labels = []

    for fname in fname_to_labels:
        filenames.append(fname)
        labels.append(frozenset(fname_to_labels[fname]))
    
    return LazyIO(images_folder, filenames, labels)

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
    labels = []

    for i in range(size):
        # Convert the image to floating point [0;1] range while we're at it.
        image = np.array(
            img[i*rows*cols:(i+1)*rows*cols],
            theano.config.floatX
        ).reshape([1, rows, cols]) / 255.
        images.append(image)
        labels.append(lbl[i])

    return ListDataset(images, labels)
    
class Dataset:
    """ A dataset of samples intended for comsumption by a convnet. 3 rules:
        - a dataset is an iterator of samples (method next()).
        - there is a finite, nonzero and known number of samples (len(dataset))
        - the samples can be iterated through in random order (method shuffle())
    """
    def shuffle(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        """ Returns the number of samples of the dataset.
        """
        raise NotImplementedError()

    def get_labels(self):
        """ Returns a list of frozenset of labels for each sample.
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
    def __init__(self, samples, labels):
        assert len(samples) > 0
        self.samples = samples
        self.labels = labels
        self.sample_shape = list(self.samples[0].shape)
        assert np.all([list(sample.shape) == self.sample_shape for sample in samples])
        self.permutation = np.array(range(len(samples)))

    def shuffle(self, permutation):
        self.permutation = permutation

    def __iter__(self):
        for i in range(self.permutation.size):
            if self.labels == None:
                yield (self.samples[self.permutation[i]],{})
            else:
                yield self.samples[self.permutation[i]]

    def __len__(self):
        return self.permutation.size

    def get_labels(self):
        # Apply the permutation to the labels.
        permut_labels = []

        for i in range(self.permutation.size):
            permut_labels.append(self.labels[self.permutation[i]])
        
        return permut_labels

class ListDataset(BaseListDataset, DatasetMixin):
    pass

class BaseLazyIO(Dataset):
    """ Lazy loading of an image dataset.
    """
    def __init__(self, folder, filenames, labels):
        assert len(filenames) > 0
        self.folder = folder
        self.filenames = filenames
        self.labels = labels
        self.permutation = np.array(range(len(filenames)))

    def __iter__(self):
        # Iterate through each filename, load it and yield the resulting image
        # (converted to floatX, [0;1] range).
        for i in range(len(self)):
            bgr_image = cv2.imread(os.path.join(
                self.folder,
                self.filenames[self.permutation[i]]
            ))
            yield bgr_image.astype(theano.config.floatX) / 255

    def __len__(self):
        return self.permutation.size

    def shuffle(self, permutation):
        self.permutation = permutation

    def get_labels(self):
        # Apply the permutation to the labels.
        permut_labels = []

        for i in range(self.permutation.size):
            permut_labels.append(self.labels[self.permutation[i]])
        
        return permut_labels

class LazyIO(BaseLazyIO, DatasetMixin):
    pass

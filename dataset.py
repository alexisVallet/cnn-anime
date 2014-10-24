import cv2
import theano
import numpy as np
import os, struct
from array import array
import cPickle as pickle

from jpeg_decode import jpeg_decode

def mini_batch_split(samples, batch_size):
    """ Splits a dataset into roughly equal sized batches of roughly a
        given size.

    Arguments:
        samples
            dataset to split.
        batch_size
            number of samples in a given batch.
    Returns:
       an array of nb_batches+1 elements indicating starting (inclusive)
       and ending (exclusive) indexes for batch i as splits[i] and splits[i+1]
       respectively.
    """
    # Determine batches by picking the number of batch which,
    # when used to divide the number of samples, best approximates
    # the desired batch size.
    flt_nb_samples = float(len(samples))
    ideal_nb_batches = flt_nb_samples / batch_size
    lower_nb_batches = np.floor(ideal_nb_batches)
    upper_nb_batches = np.ceil(ideal_nb_batches)
    lower_error = abs(flt_nb_samples / lower_nb_batches - batch_size)
    upper_error = abs(flt_nb_samples / upper_nb_batches - batch_size)
    nb_batches = (int(lower_nb_batches) if lower_error < upper_error 
                  else int(upper_nb_batches))
    # Split the dataset into that number of batches in roughly equal-sized
    # batches.
    return np.round(np.linspace(
        0, len(samples), 
        num=nb_batches+1)
    ).astype(np.int32)

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
        labels.append(str(lbl[i]))

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
        self.permutation = np.array(range(len(samples)))

    def shuffle(self, permutation):
        self.permutation = permutation

    def __iter__(self):
        for i in range(self.permutation.size):
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
        self.sample_shape = None

    def __iter__(self):
        # Iterate through each filename, load it and yield the resulting image
        # (converted to floatX, [0;1] range, in (nb_channels, rows, cols) shape).
        for i in range(len(self)):
            full_fname = os.path.join(
                self.folder,
                self.filenames[self.permutation[i]]
            )
            bgr_image = cv2.imread(full_fname)
            if bgr_image == None:
                raise ValueError("Unable to load " + repr(full_fname))
            rows, cols, nb_channels = bgr_image.shape
            bgr_image_dimshfl = np.empty(
                [nb_channels, rows, cols],
                theano.config.floatX
            )
            for j in range(nb_channels):
                bgr_image_dimshfl[j] = bgr_image[:,:,j]
                
            yield bgr_image_dimshfl / 255

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

class BaseCompressedDataset(Dataset):
    """ Datasets which loads the raw jpeg data in memory and uncompresses it on the fly.
        Useful for datasets which fit in memory when compressed.
    """
    def __init__(self, folder, filenames, labels):
        assert len(filenames) > 0
        # Load the raw jpeg data.
        self.jpeg_imgs = []
        for fname in filenames:
            with open(os.path.join(folder, fname), 'rb') as img_raw:
                self.jpeg_imgs.append(
                    np.fromfile(img_raw, np.uint8)
                )
        self.labels = labels
        self.permutation = np.array(range(len(filenames)))
        self.sample_shape = None

    def __iter__(self):
        for i in range(len(self)):
            bgr_image = jpeg_decode(self.jpeg_imgs[self.permutation[i]])
                
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

class CompressedDataset(BaseCompressedDataset, DatasetMixin):
    pass

def load_pixiv_1M(images_folder, set_pkl, dataset_class=LazyIO):
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
        fname_to_labels = pickle.load(dataset_file)
    filenames = []
    labels = []

    for fname in fname_to_labels:
        filenames.append(fname)
        labels.append(frozenset(fname_to_labels[fname]))
    
    return dataset_class(images_folder, filenames, labels)

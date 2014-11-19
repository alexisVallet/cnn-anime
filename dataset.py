import cv2
import numpy as np
import os, struct
import os.path
from array import array
import cPickle as pickle
import json

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
        image = np.array(
            img[i*rows*cols:(i+1)*rows*cols],
            np.uint8
        ).reshape([rows, cols, 1])
        images.append(image)
        labels.append(frozenset([str(lbl[i])]))

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
            np.float32
        )
        i = 0
        for sample in iter(self):
            samples_array[i] = sample
            i += 1
        return samples_array

class BaseIdentityDataset(Dataset):
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

class IdentityDataset(BaseIdentityDataset, DatasetMixin):
    pass
    
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
            cv_img = self.samples[self.permutation[i]]
            yield np.rollaxis(cv_img.astype(np.float32), 2, 0) / 255

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
            yield np.rollaxis(bgr_image.astype(np.float32), 2, 0) / 255

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
            bgr_image = cv2.imdecode(self.jpeg_imgs[self.permutation[i]], 1)
                
            yield np.rollaxis(bgr_image.astype(np.float32), 2, 0) / 255

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

def load_da_180(images_folder):
    names = [
        'amuro',
        'asuka',
        'char',
        'chirno',
        'conan',
        'jigen',
        'kouji',
        'lupin',
        'majin',
        'miku',
        'ray',
        'rufy'
    ]
    nb_img_cls = 15
    a_code = ord('a')
    images = []
    labels = []
    
    for name in names:
        for i in range(nb_img_cls):
            fname = os.path.join(
                images_folder,
                name + '_' + chr(a_code + i) + '.png'
            )
            a_image = cv2.imread(fname)
            maskname = os.path.join(
                images_folder,
                name + '_' + chr(a_code + i) + '.png-mask.png'
            )
            alpha_mask = 1 - (cv2.imread(maskname, cv2.CV_LOAD_IMAGE_GRAYSCALE).astype(np.float32) / 255)
            rows, cols = a_image.shape[0:2]
            img = (alpha_mask.reshape([rows, cols, 1]) * a_image[:,:,0:3]).astype(np.uint8)
            images.append(img)
            labels.append(frozenset([name]))
    return (images, labels)

def load_da_1000(images_folder, bb_folder=None):
    labels = [
        'asahina_mikuru',
        'asuka_langley',
        'faye_valentine',
        'ginko',
        'kaiji',
        'lain',
        'maya_fey',
        'miku_hatsune',
        'monkey_d_luffy',
        'motoko_kusanagi',
        'naoto_shirogane',
        'phoenix_wright',
        'radical_edward',
        'rei_ayanami',
        'roronoa_zoro',
        'sakura_haruno',
        'shigeru_akagi',
        'suzumiya_haruhi',
        'uzumaki_naruto',
        'yu_narukami'
    ]
    nb_folds = 5
    nb_img_cls = 50
    folds_path = os.path.join(images_folder, '5-fold')
    folds_img = []
    folds_lbl = []

    for i in range(5):
        fold_images = []
        fold_labels = []
        for label in labels:
            for j in range(nb_img_cls):
                fname = os.path.join(
                    folds_path,
                    repr(i),
                    'positives',
                    label + '_' + repr(j) + '.jpg'
                )
                if os.path.isfile(fname):
                    image = cv2.imread(fname)
                    if bb_folder != None:
                        with open(os.path.join(bb_folder, label + '_' + repr(j) + '_bb.json')) as bbfile:
                            bbox = json.load(bbfile)
                            [[x1, y1], [x2, y2]] = bbox[0]
                            image = image[y1:y2,x1:x2]
                    fold_images.append(image)
                    fold_labels.append(frozenset([label]))
        folds_img.append(fold_images)
        folds_lbl.append(fold_labels)
    return (folds_img, folds_lbl)

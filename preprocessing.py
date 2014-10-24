""" Dataset preprocessing and extension. All quite overengineered I'm afraid.
    For my defense, all these were written in (verbose, redundant) classes just
    so they would stay picklable.
"""
from dataset import Dataset, DatasetMixin
import cv2
import numpy as np
import theano

class DatasetTransformer:
    """ A dataset transformer transforms datasets (duh) for both training and
        testing, under the condition that the test transform is a functor in
        the category of pure functions (i.e. it is a sample-wise transform)
        taking as additional input info from training data.

        Note:
        train_data_transform should always be called once before test_data_transform,
        as the transformer object will track some state it learnt from the training
        data. In this sense, the dataset transform should be considered a part of the
        training algorithm.
    """
    def train_data_transform(self, train_set):
        """ Transforms a dataset for training purposes.
        """
        raise NotImplementedError()

    def test_data_transform(self, test_set):
        """ Transform a test (or validation) set.
        """
        raise NotImplementedError()

class MeanSubtraction(DatasetTransformer):
    """ Dataset transformer for mean pixel value subtraction.
    """
    def __init__(self, nb_channels):
        self.nb_channels = nb_channels
    
    class BaseTrainMeanSubtraction(Dataset):
        def __init__(self, nb_channels, dataset):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            # First accumulate the mean pixel value.
            self.mean_pixel = np.zeros(
                [nb_channels, 1, 1],
                np.float64
            )
            nb_samples = len(self.dataset)
            
            for image in self.dataset:
                new_shape = ([nb_channels, np.prod(image.shape[1:])])
                image_mean = np.mean(
                    image.reshape(new_shape),
                    axis=1
                ).astype(np.float64).reshape([nb_channels, 1, 1])
                self.mean_pixel += image_mean / nb_samples
            
            self.mean_pixel = self.mean_pixel.astype(theano.config.floatX)
    
        def __iter__(self):
            # Then subtract it from each image.
            for image in self.dataset:
                yield image - self.mean_pixel

        def __len__(self):
            return len(self.dataset)
    
        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)

        def get_labels(self):
            return self.dataset.get_labels()

    class TrainMeanSubtraction(BaseTrainMeanSubtraction, DatasetMixin):
        pass

    def train_data_transform(self, dataset):
        self.trainset = self.TrainMeanSubtraction(self.nb_channels, dataset)

        return self.trainset

    class BaseTestMeanSubtraction(Dataset):
        def __init__(self, train_mean, dataset):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            self.train_mean = train_mean

        def __iter__(self):
            # Just subtract the training mean.
            for image in self.dataset:
                yield image - self.train_mean

        def __len__(self):
            return len(self.dataset)

        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)
        
        def get_labels(self):
            return self.dataset.get_labels()

    class TestMeanSubtraction(BaseTestMeanSubtraction, DatasetMixin):
        pass

    def test_data_transform(self, test_set):
        return self.TestMeanSubtraction(self.trainset.mean_pixel, test_set)

class SingleLabelConversion(DatasetTransformer):
    """ Dataset transformer for multi-label to single label conversion of training data.
        Used for the CNN classifier. Was a bit tricky to implement, but this does not force
        any unnecessary loading of samples. It does require tracking the labels, which takes
        4 bytes per duplicated sample.
    """
    def __init__(self):
        pass

    class BaseTrainSingleLabel(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            # Keep track of which labels (in the order of the dataset) are associated
            # to which sample in an array. This array will also be updated in case of
            # shuffling.
            multi_labels = dataset.get_labels()
            nb_labels = reduce(lambda nb, ml: nb + len(ml), multi_labels, 0)
            self.labels = []
            self.label_to_sample = np.empty(
                [nb_labels],
                np.int32
            )

            l_idx = 0
            for i in range(len(dataset)):
                nb_ml = len(multi_labels[i])
                self.labels += list(multi_labels[i])
                self.label_to_sample[l_idx:l_idx+nb_ml] = i
                l_idx += nb_ml
            # Then, "shuffle" the inner array with this "permutation" (which is not
            # really a permutation since we repeat indices). This will make the inner
            # dataset repeat the sample accordingly. We send a copy so the underlying
            # dataset does not hold a reference to the one we have here, in case it does
            # some modifications of its own.
            self.dataset.shuffle(self.label_to_sample.copy())

        def __iter__(self):
            # The inner dataset is configure to yield the right samples by invariant
            # guaranteed by the constructor and the shuffle method.
            return iter(self.dataset)

        def __len__(self):
            # The size of the dataset is the number of single labels.
            return self.label_to_sample.size
        
        def shuffle(self, permutation):
            # Shuffle the labels to sample array, before feeding it back to the inner dataset.
            self.label_to_sample = self.label_to_sample[permutation]
            self.dataset.shuffle(self.label_to_sample.copy())
            # Also take care of the labels.
            new_labels = []
            for i in range(len(self.labels)):
                new_labels.append(self.labels[permutation[i]])
            self.labels = new_labels

        def get_labels(self):
            # The labels are also the one of the inner dataset, flattened into single labels.
            return self.labels

    class TrainSingleLabel(BaseTrainSingleLabel, DatasetMixin):
        pass

    def train_data_transform(self, dataset):
        return self.TrainSingleLabel(dataset)

    def test_data_transform(self, dataset):
        # Doesn't do anything.

        return dataset

def center_patch(patch_size, image):
    assert len(image.shape) == 3
    # take the center patch.
    patch = None
    rows, cols, nb_channels = image.shape
    if rows < cols:
        pad_left = (cols - rows) // 2
        patch = image[:, pad_left:pad_left+rows, :]
    if cols <= rows:
        pad_top = (rows - cols) // 2
        patch =  image[pad_top:pad_top+cols, :, :]
    patch_resized = cv2.resize(patch, (patch_size, patch_size))

    return np.rollaxis(patch_resized, 2, 0) # switch to c01

class FixedPatches(DatasetTransformer):
    """ Picks a fixed, resized patch from the middle of the image. Does not actually extend
        the dataset. Takes as inputs 01c images (as outputted by OpenCV), outputs c01 images
        (as required by Theano).
    """
    def __init__(self, nb_channels, patch_size):
        self.nb_channels = nb_channels
        self.patch_size = patch_size

    class BaseFixedPatchesSet(Dataset):
        # This dataset transform does exactly the same thing to both training
        # and test data, yay!
        def __init__(self, nb_channels, patch_size, dataset):
            self.nb_channels = nb_channels
            self.patch_size = patch_size
            self.sample_shape = [nb_channels, patch_size, patch_size]
            self.dataset = dataset

        def __iter__(self):
            for image in self.dataset:
                yield center_patch(self.patch_size, image)

        def __len__(self):
            return len(self.dataset)

        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)
        
        def get_labels(self):
            return self.dataset.get_labels()

    class FixedPatchesSet(BaseFixedPatchesSet, DatasetMixin):
        pass

    def train_data_transform(self, dataset):
        return self.FixedPatchesSet(self.nb_channels, self.patch_size, dataset)
    
    def test_data_transform(self, dataset):
        return self.FixedPatchesSet(self.nb_channels, self.patch_size, dataset)

class NameLabels(DatasetTransformer):
    """ Converts arbitrary hashable labels to int labels.
    """

    def __init__(self):
        self.label_to_int = None

    class BaseNameLabelsSet(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            self.int_to_label = list(
                reduce(
                    lambda lset, l: lset.union(l) if isinstance(l, frozenset) else lset.union([l]),
                    dataset.get_labels(),
                    frozenset()
                )
            )
            self.label_to_int = {}
            for i in range(len(self.int_to_label)):
                self.label_to_int[self.int_to_label[i]] = i
    
        def __iter__(self):
            return iter(self.dataset)

        def __len__(self):
            return len(self.dataset)

        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)

        def get_labels(self):
            def names_to_int(names):
                if isinstance(names, str):
                    return self.label_to_int[names]
                elif isinstance(names, frozenset):
                    return frozenset(map(
                        lambda name: self.label_to_int[name],
                        names
                    ))
            
            return map(
                names_to_int,
                self.dataset.get_labels()
            )

    class NameLabelsSet(BaseNameLabelsSet, DatasetMixin):
        pass

    class BaseTestNameLabelsSet(Dataset):
        def __init__(self, dataset, label_to_int):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            self.label_to_int = label_to_int
    
        def __iter__(self):
            return iter(self.dataset)

        def __len__(self):
            return len(self.dataset)

        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)
        
        def get_labels(self):
            def names_to_int(names):
                if isinstance(names, str):
                    return self.label_to_int[names]
                elif isinstance(names, frozenset):
                    return frozenset(map(
                        lambda name: self.label_to_int[name],
                        names
                    ))
            
            return map(
                names_to_int,
                self.dataset.get_labels()
            )

    class TestNameLabelsSet(BaseTestNameLabelsSet, DatasetMixin):
        pass
        
    def train_data_transform(self, dataset):
        new_set = self.NameLabelsSet(dataset)
        self.label_to_int = new_set.label_to_int
        self.int_to_label = new_set.int_to_label

        return new_set

    def test_data_transform(self, test_data):
        return self.TestNameLabelsSet(test_data, self.label_to_int)

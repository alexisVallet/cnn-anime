""" Dataset preprocessing and extension.
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
    """
    def train_data_transform(self, dataset):
        """ Transforms a dataset for training purposes.
        """
        raise NotImplementedError()

    def test_sample_transform(self, sample):
        """ Transform a single sample for test purposes.
        """
        raise NotImplementedError()

class DatasetTransformerMixin:
    class BaseTestDataset(Dataset):
        def __init__(self, parent, dataset):
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
            self.parent = parent

        def __iter__(self):
            for sample in self.dataset:
                yield self.parent.test_sample_transform(sample)

        def __len__(self):
            return len(self.dataset)

        def shuffle(self, permutation):
            self.dataset.shuffle(permutation)

        def get_labels(self):
            return self.dataset.get_labels()

    class TestDataset(BaseTestDataset, DatasetMixin):
        pass
    
    def test_data_transform(self, dataset):
        return self.TestDataset(self, dataset)

class BaseMeanSubtraction(DatasetTransformer):
    """ Dataset transformer for mean pixel value subtraction.
    """
    def __init__(self):
        pass
    
    class BaseTrainMeanSubtraction(Dataset):
        def __init__(self, dataset):
            assert len(dataset.sample_shape) == 3
            self.dataset = dataset
            self.sample_shape = dataset.sample_shape
    
        def __iter__(self):
            # First accumulate the mean pixel value.
            self.mean_pixel = np.zeros(
                [self.dataset.sample_shape[0], 1, 1],
                np.float64
            )
            nb_samples = len(self.dataset)
            new_shape = ([self.dataset.sample_shape[0], 
                        np.prod(self.dataset.sample_shape[1:])])
            
            for image in self.dataset:
                self.mean_pixel += np.mean(
                    image.reshape(new_shape),
                    axis=1,
                    keepdims=True
                ).astype(np.float64) / nb_samples
            
            self.mean_pixel = self.mean_pixel.astype(theano.config.floatX)
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
        self.trainset = self.TrainMeanSubtraction(dataset)

        return self.trainset

    def test_sample_transform(self, sample):
        """ The test transform simply subtracts the mean of the training data to the test
            sample.
        """
        return sample - self.trainset.mean_pixel
    
class MeanSubtraction(BaseMeanSubtraction, DatasetTransformerMixin):
    pass

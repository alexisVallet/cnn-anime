""" Dataset preprocessing and extension. All quite overengineered I'm afraid.
    For my defense, all these were written in (verbose, redundant) classes just
    so they would stay picklable.
"""
from dataset import Dataset, DatasetMixin
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import os.path
import cPickle as pickle

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

    def proba_transform(self, probas):
        """ If your dataset transformation extends the dataset, you need to transform the
            output probability matrix of the classifier (for a test dataset) back into the
            original size. By default does nothing, which is fine and dandy if your dataset
            does not extend.
        """
        return probas

class BaseTrainMeanSubtraction(Dataset):
    def __init__(self, nb_channels, dataset, cache_file=None):
        self.dataset = dataset
        self.sample_shape = dataset.sample_shape
        # First accumulate the mean pixel value if it has not
        # been cached yet.
        if cache_file == None or not os.path.isfile(cache_file):
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
            
            self.mean_pixel = self.mean_pixel.astype(np.float32)
            # Cache the results if possible.
            if cache_file != None:
                with open(cache_file, 'wb') as cache:
                    pickle.dump(self.mean_pixel, cache, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(cache_file, 'rb') as cache:
                self.mean_pixel = pickle.load(cache)
    
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

class MeanSubtraction(DatasetTransformer):
    """ Dataset transformer for mean pixel value subtraction.
    """
    def __init__(self, nb_channels, cache_file=None):
        self.nb_channels = nb_channels
        self.cache_file = cache_file

    def train_data_transform(self, dataset):
        self.trainset = TrainMeanSubtraction(self.nb_channels, dataset, self.cache_file)

        return self.trainset

    def test_data_transform(self, test_set):
        return TestMeanSubtraction(self.trainset.mean_pixel, test_set)

class BaseResizeSet(Dataset):
    def __init__(self, min_dim, dataset):
        self.min_dim = min_dim
        self.dataset = dataset
        self.sample_shape = None

    def __iter__(self):
        for image in self.dataset:
            rows, cols = image.shape[1:]
            n_rows, n_cols = (None, None)
            if rows > cols:
                n_rows = int(round(self.min_dim * float(rows) / cols))
                n_cols = self.min_dim
            else:
                n_cols = int(round(self.min_dim * float(cols) / rows))
                n_rows = self.min_dim
            cv_resized = resize(
                np.rollaxis(image, 0, 3),
                (n_cols, n_rows)
            )
            yield np.rollaxis(cv_resized, 2, 0)

    def __len__(self):
        return len(self.dataset)

    def get_labels(self):
        return self.dataset.get_labels()

    def shuffle(self, permutation):
        self.dataset.shuffle(permutation)

class ResizeSet(BaseResizeSet, DatasetMixin):
    pass

    
class Resize(DatasetTransformer):
    def __init__(self, min_dim):
        self.min_dim = min_dim

    def train_data_transform(self, dataset):
        return ResizeSet(self.min_dim, dataset)

    def test_data_transform(self, dataset):
        return ResizeSet(self.min_dim, dataset)

def center_patch(patch_size, image):
    assert len(image.shape) == 3
    # take the center patch.
    patch = None
    nb_channels, rows, cols = image.shape
    if rows < cols:
        pad_left = (cols - rows) // 2
        patch = image[:, :, pad_left:pad_left+rows]
    if cols <= rows:
        pad_top = (rows - cols) // 2
        patch =  image[:, pad_top:pad_top+cols, :]
    patch_resized = np.rollaxis(
        resize(np.rollaxis(patch, 0, 3), (patch_size, patch_size)),
        2,
        0
    )

    return patch_resized

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


class NameLabels(DatasetTransformer):
    """ Converts arbitrary hashable labels to int labels.
    """

    def __init__(self):
        self.label_to_int = None
        
    def train_data_transform(self, dataset):
        new_set = NameLabelsSet(dataset)
        self.label_to_int = new_set.label_to_int
        self.int_to_label = new_set.int_to_label

        return new_set

    def test_data_transform(self, test_data):
        return TestNameLabelsSet(test_data, self.label_to_int)

def random_patch(p_rows, p_cols, image):
    rows, cols = image.shape[1:]
    p_i = np.random.randint(0, max(1, rows - p_rows))
    p_j = np.random.randint(0, max(1, cols - p_cols))

    return image[:, p_i:p_i+p_rows, p_j:p_j+p_cols]

class BaseTrainRandomPatch(Dataset):
    def __init__(self, nb_channels, patch_rows, patch_cols, dataset):
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.sample_shape = [nb_channels, patch_rows, patch_cols]
        self.dataset = dataset

    def __iter__(self):
        for image in self.dataset:
            patch = random_patch(self.patch_rows, self.patch_cols, image)
            yield patch

    def shuffle(self, permutation):
        self.dataset.shuffle(permutation)

    def get_labels(self):
        return self.dataset.get_labels()

    def __len__(self):
        return len(self.dataset)

class TrainRandomPatch(BaseTrainRandomPatch, DatasetMixin):
    pass
        
class BaseTestRandomPatch(Dataset):
    def __init__(self, nb_channels, patch_rows, patch_cols, nb_test, dataset):
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.nb_test = nb_test
        self.dataset = dataset
        self.inner_perm = np.repeat(range(len(dataset)), nb_test)
        self.dataset.shuffle(self.inner_perm)
        self.sample_shape = [nb_channels, patch_rows, patch_cols]

    def __iter__(self):
        for image in self.dataset:
            yield random_patch(self.patch_rows, self.patch_cols, image)

    def shuffle(self, permutation):
        self.dataset.shuffle(self.inner_perm[permutation])

    def get_labels(self):
        return self.dataset.get_labels()
        
    def __len__(self):
        return len(self.dataset)

class TestRandomPatch(BaseTestRandomPatch, DatasetMixin):
    pass
    
class RandomPatch(DatasetTransformer):
    # Picks normally distributed random patches (i.e. gaussian with mean the center
    # of the image, stretched to fit the aspect ratio) at training time.
    
    def __init__(self, nb_channels, patch_rows, patch_cols, prediction=('rand_subwin', 10)):
        self.nb_channels = nb_channels
        self.patch_rows = patch_rows
        self.patch_cols = patch_cols
        self.prediction = prediction
                
    def train_data_transform(self, dataset):        
        return TrainRandomPatch(self.nb_channels, self.patch_rows,
                                self.patch_cols, dataset)

    def test_data_transform(self, dataset):
        if self.prediction == 'full_size':
            return dataset
        else:
            return TestRandomPatch(self.nb_channels, self.patch_rows,
                                   self.patch_cols, self.prediction[1], dataset)
    
    def proba_transform(self, probas):
        if self.prediction == 'full_size':
            return probas
        else:
            nb_samples = probas.shape[0]
            nb_test = self.prediction[1]
            assert nb_samples % nb_test == 0
            # Averages probas for the nb_test samples from which random patches were drawn.
            out_probas = np.empty(
                [nb_samples / nb_test, probas.shape[1]],
                np.float32
            )
            for i in range(nb_samples / nb_test):
                out_probas[i] = np.mean(probas[i*nb_test:(i+1)*nb_test], axis=0)

            return out_probas

class BaseRandomFlipSet(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.sample_shape = dataset.sample_shape

    def __iter__(self):
        for image in self.dataset:
            flip = np.random.uniform(0, 1)
            flipped = None
            if flip <= 0.5:
                flipped = image[:,:,::-1]
            else:
                flipped = image
            yield flipped

    def shuffle(self, permutation):
        self.dataset.shuffle(permutation)

    def get_labels(self):
        return self.dataset.get_labels()

    def __len__(self):
        return len(self.dataset)

class RandomFlipSet(BaseRandomFlipSet, DatasetMixin):
    pass
        
class RandomFlip(DatasetTransformer):
    # Randomly flips training samples horizontally.
    def __init__(self):
        pass

    def train_data_transform(self, dataset):
        return RandomFlipSet(dataset)

    def test_data_transform(self, dataset):
        return dataset


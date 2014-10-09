import theano
import theano.tensor as T
import numpy as np

import classifier

class BaseCNN:
    def __init__(self, conv_mp_layers, fc_layers):
        """ Initializes a convolutional neural net with a specific architecture.
            Uses ReLU non-linearities, and dropout regularization.

        Arguments:
            conv_layers
                list of layer objects specifying the architecture of the
                convolution and max-pooling layers.
            fc_layers
                list of ints specifying the width of each final fully
                connected layer.
            conv_init
                initialization procedure for convolutional filters.
            fc_init
                initialization procedure for fully-connected layer weights.
        """
        assert np.all([isinstance(l, ConvLayer) or isinstance(l, MaxPoolLayer)
                       for l in conv_mp_layers])
        assert np.all([isinstance(l, FCLayer) for l in fc_layers])
        self.layers = CombineLayers(conv_mp_layers + fc_layers + [SoftmaxLayer])

    def cost_function(self, batch):
        """ Returns the symbolic cost function of the convolutional neural
            net.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
        """

class Layer:
    def forward_pass(self, batch):
        """ Returns symbolic output feature maps from the input symbolic
            feature maps. The output feature maps should represent the computation
            of the layer, and should be automatically differentiable.

        Arguments:
            fmaps
                input theano symbolic feature maps, in a 4D tensor with shape
                (batch_size, in_dim, nb_rows, nb_cols).
        Returns:
            An output theano symbolic tensor.
        """
        raise NotImplementedError()

    def __add__(self, other):
        assert isinstance(other, Layer)

class CombineLayers(Layer):
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, batch):
        computation = batch
        
        for l in self.layers:
            computation = l.forward_pass(computation)

        return computation

class SoftmaxLayer(Layer):
    """ Softmax layer.
    """
    
class FCLayer(Layer):
    """ Fully connected layer with ReLU non-linearity.
    """
    
    def __init__(self, init_weights, init_biases):
        """ Initializes the fully-connected layer with a matrix of
            initial weights.

        Arguments:
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        self.weights = theano.shared(init_weights)
        self.biases = theano.shared(init_biases)

    def forward_pass(self, input_vector):
        return T.maxmium(0, T.dot(input_vector, self.weights) + self.biases)

class ConvLayer(Layer):
    """ Convolutional layer with ReLU non-linearity.
    """
    
    def __init__(self, init_filters, init_biases, fmaps_shape=None):
        """ Initializes a convolutional layer with a set of initial filters and biases.

        Arguments:
            init_filters
                4D numpy array of shape (nb_filters, nb_features, nb_rows, nb_cols).
            init_biases
                numpy vector of nb_filters elements for initial bias values.
            fmap_shape
                shape of the input feature map, optional. Used for internal theano
                optimizations.
        """
        assert len(init_filters.shape) == 4
        assert len(init_filters.shape) == 1
        # Store the initial filters in a shared variable.
        self.filters = theano.shared(init_filters)
        self.biases = theano.shared(init_biases)
        self.filter_shape = init_filters.shape
        self.fmaps_shape = fmaps.shape

    def forward_pass(self, fmaps):
        # Computes the raw convolution output.
        out_fmaps = T.nnet.conv.conv2d(
            fmaps,
            self.filters,
            filter_shape=self.filter_shape,
            image_shape=self.fmaps_shape
        )
        # Add biases and apply ReLU non-linearity.
        return T.maximum(0, out_fmaps + self.biases)

class MaxPoolLayer(Layer):
    """ Non-overlapping max pooling layer.
    """
    def __init__(self, pooling_size):
        """ Initialize a max-pooling layer with a given pooling window size.

        Arguments:
            pooling_size
                size of the max pooling window.
        """
        self.pooling_size = pooling_size

    def forward_pass(self, fmaps):
        return T.signal.downsample.max_pool_2d(fmaps, (self.pooling_size, self.pooling_size))

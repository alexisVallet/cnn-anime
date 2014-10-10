import theano
import theano.tensor as T
import numpy as np

import classifier
import optimize

class BaseCNNClassifier:
    """ Image classifier based on a convolutional neural network.
    """
    def __init__(self, architecture, optimizer, init='random', nb_channels=3):
        """ Initializes a convolutional neural network with a specific
            architecture, optimization method for training and initialization
            procedure.

        Arguments:
            architecture
                list of tuples which should be either:
                - ('conv', nb_filters, nb_rows, nb_cols)
                - ('max-pool', win_size)
                - ('fc', nb_inputs, nb_neurons)
                - ('softmax', nb_inputs, nb_outputs)
            optimizer
                optimization method to use for training the network. Should be
                an instance of Optimizer.
            init
                initialization procedure for the network. Can be:
                - 'random' for random initialization of network weights, with
                  the method described by Alex Krizhevsky, 2012.
        """
        assert init in ['random']
        assert isinstance(optimizer, Optimizer)
        assert np.all([isinstance(l, Layer) for l in architecture])
        self.optimizer = optimizer
        
        if init == 'random':
            self.cnn = init_random(architecture)

    def train(self, images, labels):
        

def init_random(architecture, nb_channels, nb_classes):
    """ Initialize a convolutional neural network at random given an architecture. All
        weights are initialized by sampling a normal distribution with mean 0 and variance
        10^-2. Biases are initialized at 0.

    Arguments:
        architecture
            see the documentation for BaseCNNClassifier.
    Returns:
        a randomly initialized convolutional neural network.
    """
    layers = []
    nb_conv_mp = 0
    nb_fc = 0
    std = 0.001**2
    input_dim = nb_channels

    # Initialize convolutional, max-pooling and FC layers.
    for layer_arch in architecture:
        if layer_arch[0] == 'conv':
            nb_filters, nb_rows, nb_cols = layer_arch[1:]
            filters = std * np.random.standard_normal([nb_filters, input_dim, nb_rows, nb_cols])
            biases = np.zeros([nb_filters])
            layers.append(ConvLayer(filters, biases))
            nb_conv_mp += 1
            # The input dimension of the next layer will be the number of filters of this
            # layer.
            input_dim = nb_filters
        elif layer_arch[0] == 'max-pool':
            # Max pooling layers leave the input dimension unchanged.
            pooling_size = layer_arch[1]
            layers.append(MaxPoolLayer(pooling_size))
            nb_conv_mp += 1
        elif layer_arch[0] == 'fc':
            # Fully connected layers, we assume that the number of inputs specified by the
            # user makes sense - it defines the size of the input images.
            nb_inputs, nb_neurons = layer_arch[1:]
            weights = std * np.random.standard_normal([nb_inputs, nb_neurons])
            biases = np.zeros([nb_neurons])
            layers.append(FCLayer(weights, biases))
            nb_fc += 1
        elif layer_arch[0] == 'softmax':
            nb_inputs, nb_outputs = layer_arch[1:]
            weights = std * np.random.standard_normal([nb_inputs, nb_outputs])
            biases = np.zeros([nb_outputs])
            layers.append(SoftmaxLayer(weights, biases))
        else:
            raise ValueError(repr(layer_arch) + " is not a valid layer architecture.")

    if len(layers) != nb_conv_mp + nb_fc + 1:
        raise ValueError("The architecture should finish with exactly one Softmax layer.")
        
    return CNN(layers[0:nb_conv_mp], layers[nb_conv_mp:nb_conv_mp+nb_fc], layers[-1])
                    
class CNN:
    """ Convolutional neural network.
    """
    
    def __init__(self, conv_mp_layers, fc_layers, softmax_layer):
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
        assert isinstance(softmax_layer, SoftmaxLayer)
        self.conv_mp_layers = conv_mp_layers
        self.fc_layers = fc_layers
        self.softmax_layer = softmax_layer

    def parameters(self):
        """ Returns the parameters of the convnet, as a list of shared theano
            variables.
        """
        return reduce(lambda params, l1: params + l1.parameters(),
                      self.conv_mp_layers + self.fc_layers + self.softmax_layer,
                      [])

    def forward_pass(self, batch):
        """ The forward pass of a convnet. This outputs a symbolic matrix of
            probabilities for each sample of the batch to belong to each class.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
        Returns:
            A symbolic (nb_images, nb_classes) matrix containing the output
            probabilities of the convnet classifier.
        """
        # First accumulate the convolutional layer forward passes.
        fpass = batch

        for conv_mp_layer in self.conv_map_layers:
            fpass = conv_mp_layer.forward_pass(fpass)

        # Reshape the output (batch_size, nb_features, nb_rows, nb_cols)
        # tensor into a data matrix (batch_size, nb_features * nb_rows * nb_cols)
        # suitable to be fed to the FC layers.
        fpass = T.flatten(fpass, outdim=2)

        # Then accumulate the FC layer forward passes.
        for fc_layer in self.fc_layers:
            fpass = fc_layer.forward_pass(fpass)

        # Finally, apply the final softmax layer.
        fpass = self.softmax_layer.forward_pass(fpass)

        return fpass
    
    def cost_function(self, batch, labels):
        """ Returns the symbolic cost function of the convolutional neural
            net.

        Arguments:
            batch
                symbolic theano 4D tensor for a batch of images. Should have
                shape (nb_images, nb_channels, nb_rows, nb_cols).
            labels
                symbolic theano 1D int tensor for corresponding image labels.
                Should have shape (nb_images,) . Labels should be in the {0, nb_classes-1}
                set.
        Returns:
            A symbolic scalar representing the cost of the function for the given
            batch. Should be (sub) differentiable with regards to self.parameters().
            
        """
        # The cost function basically sums the log softmaxed probabilities for the correct
        # labels. We average the results to make it insensitive to batch size.
        cost = - T.mean(T.log(self.forward_pass(batch)[:,labels]))

        return cost

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
    
    def parameters(self):
        """ Returns a list of shared theano variables corresponding to the layer parameters
            to optimize.
        """
        raise NotImplementedError()
    
class SoftmaxLayer(Layer):
    """ Softmax layer. Essentially multinomial logistic regression.
    """

    def __init__(self, init_weights, init_biases):
        """ Initializes the softmax layer with a matrix of initial
            weights and a vector of initial biases.

        Arguments:
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
            init_biases
                vector of nb_outputs initial biases.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights)
        self.biases = theano.shared(init_biases)

    def forward_pass(self, input_matrix):
        return T.nnet.softmax(T.dot(input_matrix, self.weights) + self.biases)

    def parameters(self):
        return [self.weights, self.biases]
    
class FCLayer(Layer):
    """ Fully connected layer with ReLU non-linearity.
    """
    
    def __init__(self, init_weights, init_biases):
        """ Initializes the fully-connected layer with a matrix of
            initial weights and a vector of initial biases.

        Arguments:
            init_weights
                matrix of nb_inputs by nb_outputs initial weights.
            init_biases
                vector of nb_outputs initial biases.
        """
        assert len(init_weights.shape) == 2
        assert len(init_biases.shape) == 1
        assert init_weights.shape[1] == init_biases.shape[0]
        self.nb_inputs, self.nb_outputs = init_weights.shape
        self.weights = theano.shared(init_weights)
        self.biases = theano.shared(init_biases)

    def forward_pass(self, input_matrix):
        return T.maxmium(0, T.dot(input_matrix, self.weights) + self.biases)

    def parameters(self):
        return [self.weights, self.biases]

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
        assert len(init_biases.shape) == 1
        assert init_filters.shape[0] == init_biases.shape[0]
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

    def parameters(self):
        return [self.filters, self.biases]

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

    def parameters(self):
        return []
